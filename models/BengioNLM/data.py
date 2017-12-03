import os
import torch


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

    def __init__(self, path, N):
        self.dictionary = Dictionary()
        self.ary = N
        self.train = self.tokenize(os.path.join(path, 'train.txt'), self.ary)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), self.ary)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), self.ary)

    def tokenize(self, path, n):

        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = ["*PAD*", "*PAD*"] + line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ngrams = []
            for sentence in f.read().split("\n"):
                sentence = ["*PAD*", "*PAD*"] + sentence.split() + ["<eos>"]
                for i, word in enumerate(sentence):
                    if i > n - 2:
                        # print("target: ", sentence[i])
                        # print("target_tensor", torch.LongTensor([self.dictionary.word_to_ix[sentence[i]]]))
                        # print("context: ", sentence[i - n + 1])
                        # print("context_tensor", torch.LongTensor([self.dictionary.word_to_ix[word] for word in sentence[i - n + 1:i]]))
                        target = torch.LongTensor([self.dictionary.word_to_ix[sentence[i]]])
                        context = torch.LongTensor([self.dictionary.word_to_ix[word] for word in sentence[i - n + 1:i]])
                        ngrams.append((context, target))

        return ngrams
