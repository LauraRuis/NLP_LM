import torch.nn as nn
import torch
from helpers import *
import torch.autograd as autograd
from data import Corpus
import math
import numpy as np


def predict_word(corpus, model, context, cuda):

    # disable dropout
    model.eval()

    word_to_ix = corpus.dictionary.word_to_ix
    context = torch.cuda.LongTensor([word_to_ix[context]])
    latent = model.init_states(cuda, 1)
    context = autograd.Variable(context)
    latent, log_probs, _, _ = model(context, latent)
    max = log_probs.view(-1).max()
    idx = 0
    for i, el in enumerate(log_probs.view(-1)):
        if el.data[0] == max.data[0]:
            idx = i
    return corpus.dictionary.ix_to_word[idx]


def generate(corpus, cuda, temperature, words, log_interval, bsz):
    ntokens = len(corpus.dictionary)
    latent = model.init_states(cuda, bsz)

    input = autograd.Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)

    if cuda:
        input.data = input.data.cuda()

    generated = []
    for i in range(words):
        latent, output, _, _ = model(input, latent)
        word_weights = output.squeeze().data.div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.ix_to_word[word_idx]

        generated.append(word + ('\n' if i % 20 == 19 else ' '))

        if i % log_interval == 0:
            print('| Generated {}/{} words'.format(i, words))
    print(''.join(generated))


def word_dependencies(model, sentence, corpus, norm, cuda):

    i_gates = []
    f_gates = []

    latent = model.init_states(cuda, 1)  # bsz = 1

    model.eval()

    # first get all input gates and forget gates per word in sentence
    for word in sentence:
        context = autograd.Variable(torch.cuda.LongTensor([corpus.dictionary.word_to_ix[word]]))
        if cuda:
            context.cuda()
        latent, log_probs, i_gate, f_gate = model(context, latent)
        i_gates.append(i_gate.data.cpu().numpy())
        f_gates.append(f_gate.data.cpu().numpy())

    important_idxs = []

    # loop over words
    for i, word in enumerate(sentence):

        # if word has a history
        if i > 0:

            # list to save weights of words in history
            weights = []

            # loop over its history
            for j in range(i):

                # multiply all subsequent forget gates
                forget_gates = f_gates[j + 1]
                for h in range(j + 2, i + 1):
                    forget_gates = np.multiply(forget_gates, f_gates[h])

                weight = np.multiply(forget_gates, i_gates[j])
                if norm == "max":
                    weights.append(np.max(weight))
                elif norm == "sum":
                    weights.append(np.sum(weight))
                # elif norm == "l2":
                #     weights.append()
            important_word_idx = weights.index(max(weights))
            important_idxs.append(important_word_idx)

    print(important_idxs)


if __name__ == "__main__":

    ####################################################################################################################
    # parameters to change
    ####################################################################################################################
    BSZ = 20
    CUDA = True
    DATA_FILE = "../../data/penn/"
    TANH_ACT = False
    BPTT = 35
    model_working_weight_analysis = "NNs/hidden_650-embed_650_drop_0.5_layers_2_tying_True.pt"
    model = torch.load(model_working_weight_analysis)
    loss_function = nn.CrossEntropyLoss()
    GET_PERPLEXITY = True
    PREDICT_WORD = False
    context = "he is a"
    GENERATE = True
    number_words = 300
    temp = 1
    ####################################################################################################################

    # get test data
    corpus = Corpus(DATA_FILE, CUDA)
    vocab_size = len(corpus.dictionary)
    print("V", vocab_size)
    test_data = batchify(corpus.test, BSZ, CUDA)

    if GET_PERPLEXITY:
        print("Perplexity")
        print(math.exp(evaluate(model, corpus, loss_function, test_data, CUDA, BSZ, BPTT)))
    elif PREDICT_WORD:
        print("Context:")
        print(context)
        print("Next word:")
        print(predict_word(corpus, model, context, CUDA))
    elif GENERATE:
        generate(corpus, CUDA, temp, number_words, 1, 1)
    else:
        print("Set either GET_PERPLEXITY, PREDICT_WORD or GENERATE to true")

    test_sentence1 = "an earthquake struck northern california killing more than N people"
    test_sentence2 = "he sits down at the piano and plays"
    test_sentence3 = "conservative party fails to secure a majority resulting in a hung parliament"

    print(test_sentence1)
    word_dependencies(model, test_sentence1.split(), corpus, "sum", True)
    print(test_sentence2)
    word_dependencies(model, test_sentence2.split(), corpus, "sum", True)
    print(test_sentence3)
    word_dependencies(model, test_sentence3.split(), corpus, "sum", True)
