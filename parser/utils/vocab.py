# -*- coding: utf-8 -*-

from collections import Counter

import torch
from pytorch_pretrained_bert import BertTokenizer


class Vocab(object):
    pad = '<pad>'
    unk = '<unk>'
    bos = '<bos>'
    eos = '<eos>'

    def __init__(self, bert_vocab, words, pos_tags, dep_tags, rels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.pad, self.unk, self.bos] + sorted(words)
        self.pos_tags = [self.bos] + sorted(pos_tags)
        self.dep_tags = [self.bos] + sorted(dep_tags)
        self.rels = [self.bos] + sorted(rels)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.pos_tag_dict = {tag: i for i, tag in enumerate(self.pos_tags)}
        self.dep_tag_dict = {tag: i for i, tag in enumerate(self.dep_tags)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}

        self.tokenizer = BertTokenizer.from_pretrained(bert_vocab)

        self.n_words = len(self.words)
        self.n_pos_tags = len(self.pos_tags)
        self.n_dep_tags = len(self.dep_tags)
        self.n_rels = len(self.rels)
        self.n_init = self.n_words

    def __repr__(self):
        s = f"{self.__class__.__name__}: "
        s += f"{self.n_words} words, "
        s += f"{self.n_pos_tags} pos_tags, "
        s += f"{self.n_dep_tags} dep_tags, "
        s += f"{self.n_rels} rels"

        return s

    def word2id(self, sequence):
        return torch.tensor([self.word_dict.get(word.lower(), self.unk_index)
                             for word in sequence])

    def pos_tag2id(self, sequence):
        return torch.tensor([self.pos_tag_dict.get(tag, 0)
                             for tag in sequence])

    def dep_tag2id(self, sequence):
        return torch.tensor([self.dep_tag_dict.get(tag, 0)
                             for tag in sequence])

    def rel2id(self, sequence):
        return torch.tensor([self.rel_dict.get(rel, 0)
                             for rel in sequence])

    def id2tag(self, ids):
        return [self.dep_tags[i] for i in ids]

    def id2rel(self, ids):
        return [self.rels[i] for i in ids]

    def read_embeddings(self, embed, smooth=True):
        words = [word.lower() for word in embed.tokens]
        # if the `unk` token has existed in the pretrained,
        # then replace it with a self-defined one
        if embed.unk:
            words[embed.unk_index] = self.unk

        self.extend(words)
        self.embed = torch.zeros(self.n_words, embed.dim)
        self.embed[self.word2id(words)] = embed.vectors

        if smooth:
            self.embed /= torch.std(self.embed)

    def extend(self, words):
        self.words += sorted(set(words).difference(self.word_dict))
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.n_words = len(self.words)

    def numericalize(self, corpus, dep=True, training=True):
        subwords, starts = [], []

        for seq in corpus.words:
            seq = [self.tokenizer.tokenize(token) for token in seq]
            seq = [piece if piece else ['[PAD]'] for piece in seq]
            seq = [['[CLS]']] + seq + [['[SEP]']]
            lengths = [0] + [len(piece) for piece in seq]
            # flatten the word pieces
            subwords.append(sum(seq, []))
            # record the start position of all words
            starts.append(torch.tensor(lengths).cumsum(0)[:-2])
        subwords = [torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
                    for tokens in subwords]
        mask = [torch.ones(len(tokens)) for tokens in subwords]
        start_mask = [~mask[i].byte().index_fill_(0, starts[i], 0)
                      for i in range(len(mask))]
        bert = list(zip(subwords, mask, start_mask))

        words = [self.word2id([self.bos] + seq) for seq in corpus.words]
        if not training:
            return bert, words
        if not dep:
            tags = [self.pos_tag2id([self.bos] + seq) for seq in corpus.tags]
            return bert, words, tags
        else:
            tags = [self.dep_tag2id([self.bos] + seq) for seq in corpus.tags]
            arcs = [torch.tensor([0] + seq) for seq in corpus.heads]
            rels = [self.rel2id([self.bos] + seq) for seq in corpus.rels]
            return bert, words, tags, arcs, rels

    @classmethod
    def from_corpora(cls, bert_vocab, tag_corpus, dep_corpus, min_freq=1):
        word_seqs = tag_corpus.words + dep_corpus.words
        words = Counter(word.lower() for seq in word_seqs for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        pos_tags = list({tag for seq in tag_corpus.tags for tag in seq})
        dep_tags = list({tag for seq in dep_corpus.tags for tag in seq})
        rels = list({rel for seq in dep_corpus.rels for rel in seq})
        vocab = cls(bert_vocab, words, pos_tags, dep_tags, rels)

        return vocab
