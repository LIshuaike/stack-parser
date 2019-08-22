# -*- coding: utf-8 -*-

from parser.metrics import AccuracyMethod

import torch
import torch.nn as nn


class Model(object):

    def __init__(self, config, vocab, parser):
        super(Model, self).__init__()

        self.config = config
        self.vocab = vocab
        self.parser = parser

    def train(self, loader):
        self.parser.train()

        for i, (words, chars, tags) in enumerate(loader):
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag = self.parser(words, chars)
            s_tag = s_tag[mask]
            gold_tags = tags[mask]

            loss = self.parser.get_loss(s_tag, gold_tags)
            loss = loss / self.config.update_steps
            loss.backward()
            if (i + 1) % self.config.update_steps == 0:
                nn.utils.clip_grad_norm_(self.parser.parameters(),
                                         self.config.clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self, loader):
        self.parser.eval()

        loss, metric_t = 0, AccuracyMethod()

        for words, chars, tags in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag = self.parser(words, chars)
            s_tag = s_tag[mask]
            gold_tags, pred_tags = tags[mask], s_tag.argmax(dim=-1)

            loss += self.parser.get_loss(s_tag, gold_tags)
            metric_t(pred_tags, gold_tags)
        loss /= len(loader)

        return loss, metric_t

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()

        all_tags = []
        for words, chars in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_tag = self.parser(words, chars)
            s_tag = s_tag[mask]
            pred_tags = s_tag.argmax(-1)

            all_tags.extend(torch.split(pred_tags, lens))
        all_tags = [self.vocab.id2tag(seq.tolist()) for seq in all_tags]

        return all_tags
