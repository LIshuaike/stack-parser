# -*- coding: utf-8 -*-

from parser.metrics import AccuracyMethod, AttachmentMethod

import torch
import torch.nn as nn


class Model(object):

    def __init__(self, config, vocab, parser):
        super(Model, self).__init__()

        self.config = config
        self.vocab = vocab
        self.parser = parser
        self.criterion = nn.CrossEntropyLoss()

    def train(self, loader):
        self.parser.train()

        for i, (bert, words, tags, arcs, rels) in enumerate(loader):
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag, s_arc, s_rel = self.parser(bert, words)
            s_tag, s_arc, s_rel = s_tag[mask], s_arc[mask], s_rel[mask]
            gold_tags = tags[mask]
            gold_arcs, gold_rels = arcs[mask], rels[mask]

            loss = self.get_loss(s_tag, s_arc, s_rel,
                                 gold_tags, gold_arcs, gold_rels)
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
        loss, metric_t, metric_p = 0, AccuracyMethod(), AttachmentMethod()

        for bert, words, tags, arcs, rels in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag, s_arc, s_rel = self.parser(bert, words)
            s_tag, s_arc, s_rel = s_tag[mask], s_arc[mask], s_rel[mask]
            gold_tags, pred_tags = tags[mask], s_tag.argmax(dim=-1)
            gold_arcs, gold_rels = arcs[mask], rels[mask]
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)

            loss += self.get_loss(s_tag, s_arc, s_rel,
                                  gold_tags, gold_arcs, gold_rels)
            metric_t(pred_tags, gold_tags)
            metric_p(pred_arcs, pred_rels, gold_arcs, gold_rels)
        loss /= len(loader)

        return loss, metric_t, metric_p

    @torch.no_grad()
    def predict(self, loader):
        self.parser.eval()

        all_tags, all_arcs, all_rels = [], [], []
        for bert, words in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_tag, s_arc, s_rel = self.parser(bert, words)
            s_tag, s_arc, s_rel = s_tag[mask], s_arc[mask], s_rel[mask]
            pred_tags = s_tag.argmax(dim=-1)
            pred_arcs, pred_rels = self.decode(s_arc, s_rel)

            all_tags.extend(torch.split(pred_tags, lens))
            all_arcs.extend(torch.split(pred_arcs, lens))
            all_rels.extend(torch.split(pred_rels, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        all_arcs = [seq.tolist() for seq in all_arcs]
        all_rels = [self.vocab.id2rel(seq) for seq in all_rels]

        return all_tags, all_arcs, all_rels

    def get_loss(self, s_tag, s_arc, s_rel, gold_tags, gold_arcs, gold_rels):
        s_rel = s_rel[torch.arange(len(s_rel)), gold_arcs]

        tag_loss = self.criterion(s_tag, gold_tags)
        arc_loss = self.criterion(s_arc, gold_arcs)
        rel_loss = self.criterion(s_rel, gold_rels)
        loss = tag_loss + arc_loss + rel_loss

        return loss

    def decode(self, s_arc, s_rel):
        pred_arcs = s_arc.argmax(dim=-1)
        pred_rels = s_rel[torch.arange(len(s_rel)), pred_arcs].argmax(dim=-1)

        return pred_arcs, pred_rels
