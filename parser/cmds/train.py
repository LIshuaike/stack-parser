# -*- coding: utf-8 -*-

import os
from datetime import datetime, timedelta
from parser import BiaffineParser, Model
from parser.metrics import AttachmentMethod
from parser.utils import Corpus, Embedding, Vocab
from parser.utils.data import TextDataset, batchify

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class Train(object):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--buckets', default=64, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--nopunct', dest='punct', action='store_false',
                               help='whether to exclude punctuation')
        subparser.add_argument('--ftrain', default='data/conll09/train.conllx',
                               help='path to train file')
        subparser.add_argument('--fdev', default='data/conll09/dev.conllx',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='data/conll09/test.conllx',
                               help='path to test file')
        subparser.add_argument('--fembed', default='data/giga.100.txt',
                               help='path to pretrained embedding file')
        subparser.add_argument('--unk', default=None,
                               help='unk token in pretrained embeddings')
        subparser.add_argument('--weight', action='store_true',
                               help='whether to weighted sum the layers')

        return subparser

    def __call__(self, config):
        print("Preprocess the data")
        train = Corpus.load(config.ftrain)
        dev = Corpus.load(config.fdev)
        test = Corpus.load(config.ftest)
        if config.preprocess or not os.path.exists(config.vocab):
            vocab = Vocab.from_corpus(corpus=train, min_freq=2)
            vocab.read_embeddings(Embedding.load(config.fembed, config.unk))
            torch.save(vocab, config.vocab)
        else:
            vocab = torch.load(config.vocab)
        config.update({
            'n_words': vocab.n_init,
            'n_chars': vocab.n_chars,
            'n_tags': vocab.n_tags,
            'n_rels': vocab.n_rels,
            'pad_index': vocab.pad_index,
            'unk_index': vocab.unk_index
        })
        print(vocab)

        print("Load the dataset")
        trainset = TextDataset(vocab.numericalize(train), config.buckets)
        devset = TextDataset(vocab.numericalize(dev), config.buckets)
        testset = TextDataset(vocab.numericalize(test), config.buckets)
        # set the data loaders
        train_loader = batchify(trainset,
                                config.batch_size//config.update_steps,
                                True)
        dev_loader = batchify(devset, config.batch_size)
        test_loader = batchify(testset, config.batch_size)
        print(f"{'train:':6} {len(trainset):5} sentences in total, "
              f"{len(train_loader):3} batches provided")
        print(f"{'dev:':6} {len(devset):5} sentences in total, "
              f"{len(dev_loader):3} batches provided")
        print(f"{'test:':6} {len(testset):5} sentences in total, "
              f"{len(test_loader):3} batches provided")

        print("Create the model")
        parser = BiaffineParser(config, vocab.embed).to(config.device)
        print(f"{parser}\n")

        model = Model(config, vocab, parser)

        total_time = timedelta()
        best_e, best_metric = 1, AttachmentMethod()
        model.optimizer = Adam(model.parser.parameters(),
                               config.lr,
                               (config.mu, config.nu),
                               config.epsilon)
        model.scheduler = ExponentialLR(model.optimizer,
                                        config.decay**(1/config.decay_steps))

        for epoch in range(1, config.epochs + 1):
            start = datetime.now()
            # train one epoch and update the parameters
            model.train(train_loader)

            print(f"Epoch {epoch} / {config.epochs}:")
            loss, metric_t, metric_p = model.evaluate(train_loader,
                                                      config.punct)
            print(f"{'train:':6} Loss: {loss:.4f} {metric_t} {metric_p}")
            loss, dev_metric_t, dev_metric_p = model.evaluate(dev_loader,
                                                              config.punct)
            print(f"{'dev:':6} Loss: {loss:.4f} {dev_metric_t} {dev_metric_p}")
            loss, metric_t, metric_p = model.evaluate(test_loader,
                                                      config.punct)
            print(f"{'test:':6} Loss: {loss:.4f} {metric_t} {metric_p}")

            t = datetime.now() - start
            # save the model if it is the best so far
            if dev_metric_p > best_metric and epoch > config.patience:
                best_e, best_metric = epoch, dev_metric_p
                model.parser.save(config.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= config.patience:
                break
        model.parser = BiaffineParser.load(config.model)
        loss, metric_t, metric_p = model.evaluate(test_loader, config.punct)

        print(f"max score of dev is {best_metric.score:.2%} at epoch {best_e}")
        print(f"the score of test at epoch {best_e} is {metric_p.score:.2%}")
        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")
