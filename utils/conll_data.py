from torch.utils import data
import torch
from utils import loader
import logging
import numpy as np


class BaseData(data.Dataset):

    CHAR_MAX = 8

    def __init__(self, file_dir, word_vocab, label_vocab):
        self.conll_data = self.pad_item(loader.load_conll_data_by_files(file_dir))
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.conll_data)

    def __getitem__(self, item_idx):
        data_item = self.conll_data[item_idx]
        data_word = data_item[0]
        data_tag = data_item[1]

        # raw information
        sentence_text = []
        sentence_tags = []

        # format information
        tokens_idx = []
        chars_idx = []
        tags_idx = []
        tokens_mask = []

        for word, tag in zip(data_word, data_tag):
            token_idx = self.word_vocab.get_word_id(word)
            tag_idx = self.label_vocab.get_id(tag)
            char_idx = []
            for char in word:
                char_idx.append(self.word_vocab.get_char_id(char))
            char_idx = char_idx if len(char_idx) <= self.CHAR_MAX else char_idx[:self.CHAR_MAX]
            chars_idx.append(char_idx)
            tokens_idx.append(token_idx)
            tags_idx.append(tag_idx)
            tokens_mask.append(1)
            sentence_text.append(word)
            sentence_tags.append(tag)
        sentence_len = len(tokens_idx)
        return tokens_idx, chars_idx, tags_idx, tokens_mask, sentence_len, sentence_text, sentence_tags

    @staticmethod
    def pad_item(data_item):
        conll_data = []
        for data_sentence in data_item:
            word_seq = []
            tag_seq = []
            for data_word in data_sentence:
                word_seq.append(data_word[0])
                tag_seq.append(data_word[1])
            if len(word_seq) > 0 and len(tag_seq) > 0:
                conll_data.append((word_seq, tag_seq))
        return conll_data

    @staticmethod
    def padding(batch):

        def batchify_fn(batch_idx):
            return [sample[batch_idx] for sample in batch]

        def pad_fn(batch_idx, max_length):
            return [sample[batch_idx] + [0] * (max_length - len(sample[batch_idx])) for sample in batch]

        def pad_char(max_length):
            return [[c + [0] * (BaseData.CHAR_MAX - len(c)) for c in sample[1]] +
                    [[0] * BaseData.CHAR_MAX] * (max_length - len(sample[1])) for sample in batch]

        batch = sorted(batch, key=lambda x: x[4], reverse=True)
        max_len = max(batchify_fn(4))

        tokens_idx = torch.LongTensor(pad_fn(0, max_len))
        chars_idx = torch.LongTensor(pad_char(max_len))
        tags_idx = torch.LongTensor(pad_fn(2, max_len))

        tokens_mask = torch.LongTensor(pad_fn(3, max_len))
        tokens_len = torch.LongTensor(batchify_fn(4))

        return tokens_idx, chars_idx, tags_idx, tokens_mask, tokens_len, batchify_fn(5), batchify_fn(6)


class LMData(data.Dataset):

    CHAR_MAX = 8

    def __init__(self, file_dir, word_vocab):
        self.lm_data = loader.load_lm_data_by_files(file_dir)
        self.word_vocab = word_vocab

    def __len__(self):
        return len(self.lm_data)

    def __getitem__(self, item_idx):
        data_item = self.lm_data[item_idx]

        # format information
        tokens_idx = []
        chars_idx = []
        forward_idx = []
        backward_idx = []
        tokens_mask = []

        for idx in range(len(data_item)):
            word = data_item[idx]
            token_idx = self.word_vocab.get_word_id(word)
            next_token = data_item[idx + 1] if idx + 1 < len(data_item) else '<unk>'
            forward_token_idx = self.word_vocab.get_word_id(next_token)
            before_token = data_item[idx - 1] if idx > 0 else '<unk>'
            backward_token_idx = self.word_vocab.get_word_id(before_token)

            char_idx = []
            for char in word:
                char_idx.append(self.word_vocab.get_char_id(char))
            char_idx = char_idx if len(char_idx) <= self.CHAR_MAX else char_idx[:self.CHAR_MAX]

            chars_idx.append(char_idx)
            tokens_idx.append(token_idx)
            forward_idx.append(forward_token_idx)
            backward_idx.append(backward_token_idx)
            tokens_mask.append(1)
        sentence_len = len(tokens_idx)
        if sentence_len <= 0:
            print(data_item, tokens_idx)
            exit()
        return tokens_idx, chars_idx, forward_idx, backward_idx, tokens_mask, sentence_len

    @staticmethod
    def padding(batch):

        def batchify_fn(batch_idx):
            return [sample[batch_idx] for sample in batch]

        def pad_fn(batch_idx, max_length):
            return [sample[batch_idx] + [0] * (max_length - len(sample[batch_idx])) for sample in batch]

        def pad_char(max_length):
            return [[c + [0] * (BaseData.CHAR_MAX - len(c)) for c in sample[1]] +
                    [[0] * BaseData.CHAR_MAX] * (max_length - len(sample[1])) for sample in batch]

        batch = sorted(batch, key=lambda x: x[5], reverse=True)
        max_len = max(batchify_fn(5))

        tokens_idx = torch.LongTensor(pad_fn(0, max_len))
        chars_idx = torch.LongTensor(pad_char(max_len))
        forward_idx = torch.LongTensor(pad_fn(2, max_len))
        backward_idx = torch.LongTensor(pad_fn(3, max_len))
        tokens_mask = torch.LongTensor(pad_fn(4, max_len))
        tokens_len = torch.LongTensor(batchify_fn(5))
        return tokens_idx, chars_idx, forward_idx, backward_idx, tokens_mask, tokens_len


class WordVocab:

    def __init__(self, word_embed_dim, char_embed_dim, word_lower=False, char_lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}

        self.id2char = {}
        self.char2id = {}

        self.word_lower = word_lower
        self.char_lower = char_lower

        self.word_embed_dim = word_embed_dim
        self.word_embeddings = None
        self.char_embed_dim = char_embed_dim
        self.char_embeddings = None

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

        for token in [self.pad_token, self.unk_token]:
            self.add_word(token)
            self.add_char(token)

    def __len__(self):
        return len(self.token2id)

    def add_word(self, token):

        token = token.lower() if self.word_lower else token
        if token in self.token2id:
            idx = self.token2id[token]
            self.token_cnt[token] += 1
        else:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            self.token_cnt[token] = 1

        return idx

    def add_char(self, char):
        char = char.lower() if self.char_lower else char
        if char in self.char2id:
            idx = self.char2id[char]
        else:
            idx = len(self.char2id)
            self.char2id[char] = idx
            self.id2char[idx] = char
        return idx

    def filter_tokens_by_cnt(self, min_cnt):

        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]

        self.token2id = {}
        self.id2token = {}
        for token in [self.pad_token, self.unk_token]:
            self.add_word(token)
        for token in filtered_tokens:
            self.add_word(token)

    def word_vocab_size(self):
        return len(self.token2id)

    def char_vocab_size(self):
        return len(self.char2id)

    def get_word_id(self, token):
        if token in self.token2id:
            return self.token2id[token]
        else:
            return self.token2id[self.unk_token]

    def get_char_id(self, char):
        if char in self.char2id:
            return self.char2id[char]
        else:
            return self.char2id[self.unk_token]

    def load_conll_word_vocab(self, files_dir: dict):
        for file_dir in files_dir.values():
            for token in loader.load_word_from_conll(file_dir):
                self.add_word(token)
                for char in token:
                    self.add_char(char)

    def load_lm_data_vocab(self, files_dir: dict):
        for file_dir in files_dir.values():
            for token in loader.load_word_from_lm(file_dir):
                self.add_word(token)
                for char in token:
                    self.add_char(char)

    def load_word_embeddings(self, embedding_path):
        logger = logging.getLogger("ner")
        oov_word = 0
        glove_word_embeddings = {}
        scale = np.sqrt(3.0 / self.word_embed_dim)
        with open(embedding_path, 'r', encoding="utf-8") as fin:
            for line in fin:
                contents = line.strip().split()
                token = contents[0]
                if len(contents[1:]) != self.word_embed_dim:
                    continue
                glove_word_embeddings[token] = list(map(float, contents[1:]))

        logger.info("glove embeddings size {}".format(len(glove_word_embeddings)))
        self.word_embeddings = np.zeros([self.word_vocab_size(), self.word_embed_dim])
        for token in self.token2id:
            if token in glove_word_embeddings:
                self.word_embeddings[self.get_word_id(token)] = glove_word_embeddings[token]
            elif token.lower() in glove_word_embeddings:
                self.word_embeddings[self.get_word_id(token)] = glove_word_embeddings[token.lower()]
            elif token in [self.pad_token, self.unk_token]:
                self.word_embeddings[self.get_word_id(token)] = np.zeros(self.word_embed_dim)
            else:
                self.word_embeddings[self.get_word_id(token)] = np.random.uniform(-scale, scale, self.word_embed_dim)
                oov_word += 1
        logger.info("oov rate {0}".format(float(oov_word) / self.word_vocab_size()))

    def random_char_embeddings(self):
        self.char_embeddings = np.zeros([self.char_vocab_size(), self.char_embed_dim])
        scale = np.sqrt(3.0 / self.char_embed_dim)
        for index in range(self.char_vocab_size()):
            if self.id2char[index] in [self.pad_token, self.unk_token]:
                continue
            else:
                self.char_embeddings[index, :] = np.random.uniform(-scale, scale, [1, self.char_embed_dim])

    def random_word_embeddings(self):
        self.word_embeddings = np.zeros([self.word_vocab_size(), self.word_embed_dim])
        scale = np.sqrt(3.0 / self.char_embed_dim)
        for index in range(self.word_vocab_size()):
            if self.id2token[index] in [self.pad_token, self.unk_token]:
                continue
            else:
                self.word_embeddings[index, :] = np.random.uniform(-scale, scale, [1, self.word_embed_dim])


class LabelVocab:

    def __init__(self):
        self.label2id = {}
        self.id2label = {}

        self.pad_token = '<pad>'
        self.add(self.pad_token)

    def __len__(self):
        return len(self.label2id)

    def load_label_vocab(self, files_dir: dict):
        for file_dir in files_dir.values():
            for label in loader.load_label_from_conll(file_dir):
                self.add(label)

    def load_label_vocab_by_files(self, file_dir):
        for label in loader.load_label_from_conll(file_dir):
            self.add(label)

    def add(self, label):
        if label not in self.label2id:
            idx = len(self.label2id)
            self.label2id[label] = idx
            self.id2label[idx] = label

    def get_id(self, label):
        if label in self.label2id:
            return self.label2id[label]

    def get_label(self, idx):
        if idx in self.id2label:
            return self.id2label[idx]
