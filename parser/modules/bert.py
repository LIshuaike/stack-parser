# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .scalar_mix import ScalarMix
from pytorch_pretrained_bert import BertModel


class BertEmbedding(nn.Module):

    def __init__(self, path, n_layers, n_out, dropout=0, freeze=True):
        super(BertEmbedding, self).__init__()

        self.model = BertModel.from_pretrained(path)
        self.n_layers = n_layers
        self.hidden_size = self.model.config.hidden_size

        self.scalar_mix = ScalarMix(n_layers)
        self.projection = nn.Linear(self.hidden_size, n_out, False)
        self.dropout = nn.Dropout(dropout)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, subwords, mask, start_mask):
        lens = start_mask.sum(1)
        bert, _ = self.model(subwords, attention_mask=mask)
        bert = bert[-self.n_layers:]
        bert = self.scalar_mix(bert)
        bert = pad_sequence(torch.split(bert[start_mask], lens.tolist()), True)
        bert = self.projection(bert)
        bert = self.dropout(bert)

        return bert
