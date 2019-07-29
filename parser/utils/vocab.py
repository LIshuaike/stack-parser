# -*- coding: utf-8 -*-

from collections import Counter

import torch
from pytorch_pretrained_bert import BertTokenizer


class Vocab(object):
    pad = '<pad>'
    unk = '<unk>'
    bos = '<bos>'
    eos = '<eos>'

    def __init__(self, bert_vocab, words, tags, rels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.pad, self.unk, self.bos] + sorted(words)
        self.tags = [self.bos] + sorted(tags)
        self.rels = [self.bos] + sorted(rels)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.tag_dict = {tag: i for i, tag in enumerate(self.tags)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}

        self.tokenizer = BertTokenizer.from_pretrained(bert_vocab)

        self.n_words = len(self.words)
        self.n_tags = len(self.tags)
        self.n_rels = len(self.rels)
        self.n_init = self.n_words

    def __repr__(self):
        s = f"{self.__class__.__name__}: "
        s += f"{self.n_words} words, "
        s += f"{self.n_tags} tags, "
        s += f"{self.n_rels} rels"

        return s

    def word2id(self, sequence):
        return torch.tensor([self.word_dict.get(word.lower(), self.unk_index)
                             for word in sequence])

    def tag2id(self, sequence):
        return torch.tensor([self.tag_dict.get(tag, 0)
                             for tag in sequence])

    def rel2id(self, sequence):
        return torch.tensor([self.rel_dict.get(rel, 0)
                             for rel in sequence])

    def id2tag(self, ids):
        return [self.tags[i] for i in ids]

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

    def numericalize(self, corpus, training=True):
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
        tags = [self.tag2id([self.bos] + seq) for seq in corpus.tags]
        arcs = [torch.tensor([0] + seq) for seq in corpus.heads]
        rels = [self.rel2id([self.bos] + seq) for seq in corpus.rels]

        return bert, words, tags, arcs, rels

    @classmethod
    def from_corpus(cls, bert_vocab, corpus, min_freq=1):
        words = Counter(word.lower() for seq in corpus.words for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        tags = list({tag for seq in corpus.tags for tag in seq})
        rels = list({rel for seq in corpus.rels for rel in seq})
        vocab = cls(bert_vocab, words, tags, rels)

        return vocab
