# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScalarMix(nn.Module):

    def __init__(self, n_reprs, do_layer_norm=False):
        super(ScalarMix, self).__init__()

        self.n_reprs = n_reprs
        self.do_layer_norm = do_layer_norm

        self.weights = nn.Parameter(torch.zeros(n_reprs))
        self.gamma = nn.Parameter(torch.tensor([1.0]))

    def extra_repr(self):
        s = f"n_reprs={self.n_reprs}"
        if self.do_layer_norm:
            s += f", do_layer_norm={self.do_layer_norm}"

        return s

    def forward(self, tensors, mask=None):
        normed_weights = self.weights.softmax(dim=0)

        if self.do_layer_norm:
            mask = mask.unsqueeze(-1).float()
            weighted_sum = sum(w * F.layer_norm(h * mask, h.shape)
                               for w, h in zip(normed_weights, tensors))
        else:
            weighted_sum = sum(w * h for w, h in zip(normed_weights, tensors))

        return self.gamma * weighted_sum
