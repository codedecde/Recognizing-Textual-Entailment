from __future__ import unicode_literals, print_function, division
import unicodedata
from collections import Counter
from nltk.tokenize import TweetTokenizer
import cPickle as cp
import io
import sys
import pdb

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
VOCAB_SIZE = 10000

class Lang(object):
	def __init__(self, name, lowercase = True, tokenizer = None):
		self.name = name
		self.word_count = Counter()
		self.tokenizer = tokenizer
		self.lowercase = lowercase # To lowercase all words encountered			

	def tokenize_sent(self, sentence):
		if self.tokenizer is None:						
			return sentence.split(u' ')
		else:
			return self.tokenizer.tokenize(sentence)

	def add_sentence(self,sentence):
		# sentence = sentence.encode('utf-8','replace')		
		for w in self.tokenize_sent(sentence):
			if self.lowercase:
				w = w.lower()
			self.word_count[w] += 1

	def generate_vocab(self):
		vocab = self.word_count.most_common(VOCAB_SIZE)
		self.word2ix = {"<PAD>":PAD_TOKEN, "<SOS>":SOS_TOKEN, "<EOS>":EOS_TOKEN}
		for w,_ in vocab:
			self.word2ix[w] = len(self.word2ix)		
		self.ix2word = {self.word2ix[w]:w for w in self.word2ix}

	def add_word(self, word):
		self.word2ix[word] = len(self.word2ix)
		self.ix2word[self.word2ix[word]] = word

	def __getitem__(self,item):
		if type(item) == str or type(item) == unicode:
			# Encode the string to be unicode
			item = unicode(item)
			if self.lowercase:
				item = item.lower()
			return self.word2ix[item] if item in self.word2ix else len(self.word2ix)
		else:
			return self.ix2word[item] if item in self.ix2word else "<UNK>" 
	
	def __len__(self):
		assert len(self.ix2word) == len(self.word2ix), "Index not built using generate_vocab and add_word"
		return len(self.ix2word)

	def save_file(self, filename):
		cp.dump(self.__dict__, open(filename,'wb'))
	
	def load_file(self, filename):
		self.__dict__ = cp.load(open(filename))

def build_vocab(filename, l):
	with io.open(filename, encoding='utf-8', mode='r', errors='replace') as f:
		for line in f:			
			line = line.strip().split('\t')
			l.add_sentence(line[0])
			l.add_sentence(line[1])
		l.generate_vocab()		
	return l

if __name__ == "__main__":
	# ROOT_DIR = "/home/bass/DataDir/RTE/"
	ROOT_DIR = ""
	DATA_FILE = ROOT_DIR + "data/train.txt"
	# DATA_FILE ="data/tiny_eng-fra.txt"
	l_en = Lang('en', tokenizer = TweetTokenizer())	
	l_en = build_vocab(DATA_FILE, l_en)
	save_file_name = ROOT_DIR + 'data/vocab.pkl'
	l_en.save_file(save_file_name)	





