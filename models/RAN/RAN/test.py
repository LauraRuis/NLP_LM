import os
import torch
import random

class Dictionary(object):

    def __init__(self):
        self.word_to_ix = {}
        self.ix_to_word = []

    def add_word(self, word):
        if word not in self.word_to_ix:
            self.ix_to_word.append(word)
            self.word_to_ix[word] = len(self.ix_to_word) - 1
        return self.word_to_ix[word]

    def __len__(self):
        return len(self.ix_to_word)


class Corpus(object):

    def __init__(self, path, cuda):
        self.dictionary = Dictionary()
        self.cuda = cuda
        self.path = path
        self.train = self.tokenize(os.path.join(path, 'alphabet.txt'))

    def tokenize(self, path):

        """Tokenizes a text file."""
        # assert os.path.exists(path)

        # train = open(os.path.join(self.path, 'train.txt'), 'w')
        valid = open(os.path.join(self.path, 'valid.txt'), 'w')
        # test = open(os.path.join(self.path, 'test.txt'), 'w')

        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            random.seed(1)
            for line in f:

                begin = random.randint(0, 21)
                end = random.randint(begin, 22)
                if end - begin > 2:
                    words = line.split() + ['<eos>']
                    print(begin)
                    print(end)
                    sentence = ' '.join(words[begin:end])[:-2]
                    print(sentence)
                    valid.write(sentence)
                    valid.write('\n')

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            if self.cuda:
                ids = torch.cuda.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word_to_ix[word]
                    token += 1

        return ids


if __name__ == '__main__':
    DATA_FILE = "../../data/alphabet/"

    # read data
    corpus = Corpus(DATA_FILE, True)
