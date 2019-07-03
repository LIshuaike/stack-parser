# -*- coding: utf-8 -*-

from collections import Counter

import torch
from pytorch_pretrained_bert import BertTokenizer


class Vocab(object):
    PAD = '<PAD>'
    UNK = '<UNK>'
    BOS = '<BOS>'
    EOS = '<EOS>'

    def __init__(self, bert_vocab, words, chars, pos_tags, dep_tags, rels):
        self.pad_index = 0
        self.unk_index = 1

        self.words = [self.PAD, self.UNK, self.BOS] + sorted(words)
        self.chars = [self.PAD, self.UNK, self.BOS] + sorted(chars)
        self.pos_tags = [self.BOS] + sorted(pos_tags)
        self.dep_tags = [self.BOS] + sorted(dep_tags)
        self.rels = [self.BOS] + sorted(rels)

        self.word_dict = {word: i for i, word in enumerate(self.words)}
        self.char_dict = {char: i for i, char in enumerate(self.chars)}
        self.pos_tag_dict = {tag: i for i, tag in enumerate(self.pos_tags)}
        self.dep_tag_dict = {tag: i for i, tag in enumerate(self.dep_tags)}
        self.rel_dict = {rel: i for i, rel in enumerate(self.rels)}

        self.tokenizer = BertTokenizer.from_pretrained(bert_vocab)

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_pos_tags = len(self.pos_tags)
        self.n_dep_tags = len(self.dep_tags)
        self.n_rels = len(self.rels)
        self.n_init = self.n_words

    def __repr__(self):
        info = f"{self.__class__.__name__}: "
        info += f"{self.n_words} words, "
        info += f"{self.n_chars} chars, "
        info += f"{self.n_pos_tags} pos_tags, "
        info += f"{self.n_dep_tags} dep_tags, "
        info += f"{self.n_rels} rels"

        return info

    def word2id(self, sequence):
        return torch.tensor([self.word_dict.get(word.lower(), self.unk_index)
                             for word in sequence])

    def char2id(self, sequence, max_len=20):
        char_ids = torch.zeros(len(sequence), max_len, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor([self.char_dict.get(c, self.unk_index)
                                for c in word[:max_len]])
            char_ids[i, :len(ids)] = ids

        return char_ids

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
        # if the UNK token has existed in the pretrained,
        # then use it to replace the one in the vocab
        if embed.unk:
            self.UNK = embed.unk

        self.extend(embed.tokens)
        self.embeddings = torch.zeros(self.n_words, embed.dim)

        for i, word in enumerate(self.words):
            if word in embed:
                self.embeddings[i] = embed[word]
        if smooth:
            self.embeddings /= torch.std(self.embeddings)

    def extend(self, words):
        self.words += sorted(set(words).difference(self.word_dict))
        self.chars += sorted(set(''.join(words)).difference(self.char_dict))
        self.word_dict = {w: i for i, w in enumerate(self.words)}
        self.char_dict = {c: i for i, c in enumerate(self.chars)}
        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

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
        bert = [(i, j, k) for i, j, k in zip(subwords, mask, start_mask)]

        words = [self.word2id([self.BOS] + seq) for seq in corpus.words]
        chars = [self.char2id([self.BOS] + seq) for seq in corpus.words]
        if not training:
            return bert, words, chars
        if not dep:
            tags = [self.pos_tag2id([self.BOS] + seq) for seq in corpus.tags]
            return bert, words, chars, tags
        else:
            tags = [self.dep_tag2id([self.BOS] + seq) for seq in corpus.tags]
            arcs = [torch.tensor([0] + seq) for seq in corpus.heads]
            rels = [self.rel2id([self.BOS] + seq) for seq in corpus.rels]
            return bert, words, chars, tags, arcs, rels

    @classmethod
    def from_corpora(cls, bert_vocab, tag_corpus, dep_corpus, min_freq=1):
        word_seqs = tag_corpus.words + dep_corpus.words
        words = Counter(word.lower() for seq in word_seqs for word in seq)
        words = list(word for word, freq in words.items() if freq >= min_freq)
        chars = list({char for seq in word_seqs for char in ''.join(seq)})
        pos_tags = list({tag for seq in tag_corpus.tags for tag in seq})
        dep_tags = list({tag for seq in dep_corpus.tags for tag in seq})
        rels = list({rel for seq in dep_corpus.rels for rel in seq})
        vocab = cls(bert_vocab, words, chars, pos_tags, dep_tags, rels)

        return vocab
