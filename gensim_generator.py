import Lang as Ln
from gensim.models import Word2Vec
import sys

from utils import Progbar
import io


class Generator(object):
    def __init__(self, data, n_epochs, l_en, lowercase=True):
        self.data = data
        self.epoch_number = 0
        self.model = None
        self.model_prefix = None
        self.n_epochs = n_epochs
        self.l_en = l_en
        self.lowercase = lowercase

    def __iter__(self):
        if self.model is not None:
            # Training started
            self.epoch_number += 1
            print 'STARTING EPOCH : (%d/%d)' % (self.epoch_number, self.n_epochs)
            sys.stdout.flush()
        self.bar = Progbar(len(self.data))
        for idx, line in enumerate(self.data):
            self.bar.update(idx + 1)
            line = line.lower() if self.lowercase else line
            yield self.l_en.tokenize_sent(line)
        if self.model is not None:
            if self.epoch_number != self.n_epochs:
                SAVE_FILE_NAME = self.model_prefix + '_iter_' + str(self.epoch_number) + '.model'
            else:
                # Last Epoch
                SAVE_FILE_NAME = self.model_prefix + '.model'
            self.model.save(SAVE_FILE_NAME)


ROOT_DIR = ""
if __name__ == "__main__":
    l_en = Ln.Lang('en')
    VOCAB_FILE = ROOT_DIR + 'data/vocab.pkl'
    l_en.load_file(VOCAB_FILE)
    data_file = ROOT_DIR + 'data/tinyTrain.txt'
    model_prefix = ROOT_DIR + 'gensim_models/model_all_lowercase_'

    data = []
    print 'LOADING DATA...'
    sys.stdout.flush()
    with io.open(data_file, encoding='utf-8', mode='r', errors='replace') as f:
        for line in f:
            line = line.strip().split('\t')
            data.append(line[0])
            data.append(line[1])
    print 'LOADING DONE ...'
    sys.stdout.flush()
    generator = Generator(data, 5, l_en)
    model = Word2Vec(min_count=10, iter=5, size=300, workers=5)
    model.build_vocab(generator)
    generator.model = model
    generator.model_prefix = model_prefix
    model.train(generator, total_examples=model.corpus_count, epochs=model.iter)
    print '\nVOCAB GENERATED'
