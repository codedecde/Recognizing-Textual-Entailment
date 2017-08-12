from __future__ import unicode_literals, print_function, division

from collections import Counter
from nltk.tokenize import TweetTokenizer
import cPickle as cp
import io
import pdb

import numpy as np

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
VOCAB_SIZE = 10000


class Lang(object):
    def __init__(self, name, lowercase=True, tokenizer=None):
        self.name = name
        self.word_count = Counter()
        self.character_count = Counter()
        self.tokenizer = tokenizer
        self.lowercase = lowercase  # To lowercase all words encountered
        self.embedding_matrix = None
        self.PAD_TOK_VEC = None        
        self.UNK_TOK_VEC = None        

    def tokenize_sent(self, sentence):
        if self.tokenizer is None:
            return sentence.split(u' ')
        else:
            return self.tokenizer.tokenize(sentence)

    def add_sentence(self, sentence):
        for w in self.tokenize_sent(sentence):
            if self.lowercase:
                w = w.lower()
            self.word_count[w] += 1
            for c in w:
                self.character_count[c] += 1

    def char_index(self, c):
        if self.lowercase:
            c = c.lower()
        return self.char2ix[c] if c in self.char2ix else len(self.char2ix)

    def char_vocab_size(self):
        assert len(self.ix2char) == len(self.char2ix), "Index not built using generate_vocab and add_word"
        return len(self.char2ix)

    def add_char(self, c):
        assert c not in self.char2ix, "Already present in vocab"
        self.char2ix[c] = len(self.char2ix)
        self.ix2char[self.char2ix[c]] = c

    def generate_vocab(self):
        vocab = self.word_count.most_common(VOCAB_SIZE)
        self.word2ix = {"<PAD>": PAD_TOKEN, "<SOS>": SOS_TOKEN, "<EOS>": EOS_TOKEN}
        for w, _ in vocab:
            self.word2ix[w] = len(self.word2ix)
        self.ix2word = {self.word2ix[w]: w for w in self.word2ix}
        char_vocab = self.character_count.most_common(VOCAB_SIZE)
        self.char2ix = {"<PAD>": PAD_TOKEN}
        for c,_ in char_vocab:
            self.char2ix[c] = len(self.char2ix)
        self.ix2char = {self.char2ix[c]:c for c in self.char2ix}

    def add_word(self, word, embedding=None):
        assert word not in self.word2ix, "Already present in vocab"
        self.word2ix[word] = len(self.word2ix)
        self.ix2word[self.word2ix[word]] = word
        if self.embedding_matrix is not None:
            _, n_embed = self.embedding_matrix.shape
            embedding = embedding if embedding is not None else np.random.normal(0, 1, (1, n_embed))
            self.embedding_matrix = np.concatenate([self.embedding_matrix, embedding], axis=0)

    def __getitem__(self, item):
        if type(item) == str or type(item) == unicode:
            # Encode the string to be unicode
            item = unicode(item)
            if self.lowercase:
                item = item.lower()
            return self.word2ix[item] if item in self.word2ix else len(self.word2ix)
        else:
            return self.ix2word[item] if item in self.ix2word else u"<UNK>"

    def __len__(self):
        assert len(self.ix2word) == len(self.word2ix), "Index not built using generate_vocab and add_word"
        return len(self.ix2word)

    def save_file(self, filename):
        cp.dump(self.__dict__, open(filename, 'wb'))

    def load_file(self, filename):
        self.__dict__ = cp.load(open(filename))

    def get_embedding_matrix(self):
        if self.embedding_matrix is None:
            return None
        _embedding_matrix = np.concatenate([self.PAD_TOK_VEC, self.embedding_matrix, self.UNK_TOK_VEC], axis=0)
        return _embedding_matrix


def build_vocab(filename, l):
    with io.open(filename, encoding='utf-8', mode='r', errors='replace') as f:
        for line in f:
            line = line.strip().split('\t')
            l.add_sentence(line[0])
            l.add_sentence(line[1])
        l.generate_vocab()
    return l


def build_embedding_matrix_from_gensim(l_en, gensim_model, embedding_dim=300):
    l_en.PAD_TOK_VEC = np.random.normal(0, 1, (1, embedding_dim))
    l_en.UNK_TOK_VEC = np.random.normal(0, 1, (1, embedding_dim))

    l_en.embedding_matrix = np.random.normal(0, 1, (len(l_en) - 1, embedding_dim))  # PAD TOKEN ENCODED SEPARATELY
    for w in l_en.word2ix:
        if l_en.word2ix[w] == PAD_TOKEN:
            # PAD TOKEN ENCODED SEPARATELY
            continue
        if w in gensim_model.wv:
            l_en.embedding_matrix[l_en.word2ix[w] - 1] = gensim_model.wv[w]
    return l_en


if __name__ == "__main__":
    # ROOT_DIR = "/home/bass/DataDir/RTE/"
    ROOT_DIR = ""
    DATA_FILE = ROOT_DIR + "../data/tinyTrain.txt"
    # DATA_FILE ="data/tiny_eng-fra.txt"
    l_en = Lang('en', tokenizer=TweetTokenizer())
    l_en = build_vocab(DATA_FILE, l_en)    
    save_file_name = ROOT_DIR + '../data/vocab_char.pkl'
    l_en.save_file(save_file_name)
