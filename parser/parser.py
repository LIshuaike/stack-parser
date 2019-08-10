# -*- coding: utf-8 -*-

from parser.modules import (MLP, BertEmbedding, Biaffine, BiLSTM,
                            IndependentDropout, ScalarMix, SharedDropout)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiaffineParser(nn.Module):

    def __init__(self, config, embed):
        super(BiaffineParser, self).__init__()

        self.config = config
        # the embedding layer
        self.pretrained = nn.Embedding.from_pretrained(embed)
        self.word_embed = nn.Embedding(num_embeddings=config.n_words,
                                       embedding_dim=config.n_embed)
        self.bert_embed = BertEmbedding(path=config.bert_path,
                                        n_layers=config.n_bert_layers,
                                        n_out=config.n_embed)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        self.tag_lstm = BiLSTM(input_size=config.n_embed*2,
                               hidden_size=config.n_lstm_hidden,
                               num_layers=config.n_lstm_layers,
                               dropout=config.lstm_dropout)
        self.dep_lstm = BiLSTM(
            input_size=config.n_embed*2+config.n_mlp_arc,
            hidden_size=config.n_lstm_hidden,
            num_layers=config.n_lstm_layers,
            dropout=config.lstm_dropout
        )
        if config.weight:
            self.tag_mix = ScalarMix(n_layers=config.n_lstm_layers)
            self.dep_mix = ScalarMix(n_layers=config.n_lstm_layers)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        # the MLP layers
        self.mlp_tag = MLP(n_in=config.n_lstm_hidden*2,
                           n_hidden=config.n_mlp_arc,
                           dropout=0.5)
        self.mlp_dep = MLP(n_in=config.n_lstm_hidden*2,
                           n_hidden=config.n_mlp_arc,
                           dropout=config.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_arc,
                             dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.n_lstm_hidden*2,
                             n_hidden=config.n_mlp_rel,
                             dropout=config.mlp_dropout)

        self.ffn_pos_tag = nn.Linear(config.n_mlp_arc,
                                     config.n_pos_tags)
        self.ffn_dep_tag = nn.Linear(config.n_mlp_arc,
                                     config.n_dep_tags)
        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=config.n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=config.n_mlp_rel,
                                 n_out=config.n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.weight = config.weight
        self.pad_index = config.pad_index
        self.unk_index = config.unk_index

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.word_embed.weight)
        nn.init.orthogonal_(self.ffn_pos_tag.weight)
        nn.init.orthogonal_(self.ffn_dep_tag.weight)
        nn.init.zeros_(self.ffn_pos_tag.bias)
        nn.init.zeros_(self.ffn_dep_tag.bias)

    def forward(self, bert, words, dep=True):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.word_embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.pretrained(words) + self.word_embed(ext_words)
        word_embed = word_embed[:, :max(lens)]
        bert_embed = self.bert_embed(*bert)
        embeds = self.embed_dropout(word_embed, bert_embed)
        embed = torch.cat(embeds, dim=-1)

        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()
        x = pack_padded_sequence(embed[indices], sorted_lens, True)
        if self.weight:
            x = [pad_packed_sequence(i, True)[0] for i in self.tag_lstm(x)]
            x_tag = self.lstm_dropout(self.tag_mix(x))[inverse_indices]
        else:
            x = pad_packed_sequence(self.tag_lstm(x)[-1], True)[0]
            x = self.lstm_dropout(x)[inverse_indices]
            x_tag = x
        x_tag = self.mlp_tag(x_tag)

        if not dep:
            return self.ffn_pos_tag(x_tag)

        if self.weight:
            x_dep = self.lstm_dropout(self.dep_mix(x))[inverse_indices]
        else:
            x_dep = x
        x_dep = self.mlp_dep(x_dep)

        x = torch.cat((embed, x_dep), dim=-1)
        x = pack_padded_sequence(x[indices], sorted_lens, True)
        x = self.dep_lstm(x)[-1]
        x, _ = pad_packed_sequence(x, True)
        x_dep = self.lstm_dropout(x)[inverse_indices]

        # apply MLPs to the BiLSTM output states
        arc_h = self.mlp_arc_h(x_dep)
        arc_d = self.mlp_arc_d(x_dep)
        rel_h = self.mlp_rel_h(x_dep)
        rel_d = self.mlp_rel_d(x_dep)

        s_tag = self.ffn_dep_tag(x_tag)
        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_tag, s_arc, s_rel

    @classmethod
    def load(cls, fname):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'], state['embed'])
        parser.load_state_dict(state['state_dict'])
        parser.to(device)

        return parser

    @classmethod
    def load_checkpoint(cls, fname):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(fname, map_location=device)

        return state

    def save(self, fname):
        state = {
            'config': self.config,
            'embed': self.pretrained.weight,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)

    def save_checkpoint(self, fname, epoch, optimizer, scheduler):
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(state, fname)
