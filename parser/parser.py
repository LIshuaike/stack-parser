# -*- coding: utf-8 -*-

from parser.modules import (CHAR_LSTM, MLP, BiLSTM, IndependentDropout,
                            ScalarMix, SharedDropout)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class BiaffineParser(nn.Module):

    def __init__(self, config, embed):
        super(BiaffineParser, self).__init__()

        self.config = config
        # the embedding layer
        self.pretrained = nn.Embedding.from_pretrained(embed)
        self.word_embed = nn.Embedding(num_embeddings=config.n_words,
                                       embedding_dim=config.n_embed)
        # the char-lstm layer
        self.char_lstm = CHAR_LSTM(n_chars=config.n_chars,
                                   n_embed=config.n_char_embed,
                                   n_out=config.n_embed)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        self.tag_lstm = BiLSTM(input_size=config.n_embed*2,
                               hidden_size=config.n_lstm_hidden,
                               num_layers=config.n_lstm_layers,
                               dropout=config.lstm_dropout)
        self.dep_lstm = BiLSTM(input_size=config.n_embed*2+config.n_mlp_arc,
                               hidden_size=config.n_lstm_hidden,
                               num_layers=config.n_lstm_layers,
                               dropout=config.lstm_dropout)
        if config.weight:
            self.tag_mix = ScalarMix(n_layers=config.n_lstm_layers)
            self.dep_mix = ScalarMix(n_layers=config.n_lstm_layers)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        # the MLP layers
        self.mlp_tag = MLP(n_in=config.n_lstm_hidden*2,
                           n_hidden=config.n_mlp_arc,
                           dropout=0.5)

        self.ffn_tag = nn.Linear(config.n_mlp_arc,
                                 config.n_tags)
        self.weight = config.weight
        self.pad_index = config.pad_index
        self.unk_index = config.unk_index
        self.criterion = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.word_embed.weight)
        nn.init.orthogonal_(self.ffn_tag.weight)
        nn.init.zeros_(self.ffn_tag.bias)

    def forward(self, words, chars):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.word_embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        word_embed = self.pretrained(words) + self.word_embed(ext_words)
        char_embed = self.char_lstm(chars[mask])
        char_embed = pad_sequence(torch.split(char_embed, lens.tolist()), True)
        word_embed, char_embed = self.embed_dropout(word_embed, char_embed)
        # concatenate the word and char representations
        embed = torch.cat((word_embed, char_embed), dim=-1)

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
        s_tag = self.ffn_tag(x_tag)

        return s_tag

    @classmethod
    def load(cls, fname):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'], state['embed'])
        parser.load_state_dict(state['state_dict'])
        parser.to(device)

        return parser

    def save(self, fname):
        state = {
            'config': self.config,
            'embed': self.pretrained.weight,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)

    def get_loss(self, s_tag, gold_tags):
        return self.criterion(s_tag, gold_tags)
