# from NNs.ppl41.RAN import *
import torch.nn as nn
from helpers import *
import json
import torch.autograd as autograd
from data import Corpus
import math
import torch.nn.functional as F


def predict_word(corpus, model, context, cuda, tanh_act):

    word_to_ix = corpus.dictionary.word_to_ix
    context = torch.cuda.LongTensor([word_to_ix[context]])
    hidden = model.initVars(cuda, 1)
    latent = model.initVars(cuda, 1)
    content = model.initVars(cuda, 1)
    context = autograd.Variable(context)
    if tanh_act:
        content, latent, hidden, log_probs = model(context, hidden, latent, content)
    else:
        latent, hidden, log_probs = model(context, hidden, latent)
    max = log_probs.view(-1).max()
    idx = 0
    for i, el in enumerate(log_probs.view(-1)):
        if el.data[0] == max.data[0]:
            idx = i
    return corpus.dictionary.ix_to_word[idx]


def evaluate(model, corpus, criterion, data_source, cuda, bsz, bptt, tanh_act):
    # Turn on evaluation mode which disables dropout.
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.initVars(cuda, bsz)
    latent = model.initVars(cuda, bsz)
    content = model.initVars(cuda, bsz)

    for i in range(0, data_source.size(0) - 1, bptt):

        context, target = get_batch(data_source, i, bptt, evaluation=True)
        if target.size(0) == bsz * bptt:

            if tanh_act:
                content, latent, hidden, log_probs = model(context, hidden, latent, content)
            else:
                latent, hidden, log_probs = model(context, hidden, latent)

            output_flat = log_probs.view(-1, ntokens)
            # print(output_flat)
            # print(target)
            total_loss += len(context) * criterion(output_flat, target).data
            hidden = repackage_hidden(hidden)

    return total_loss[0] / len(data_source)


def generate(corpus, cuda, temperature, words, log_interval, bsz, tanh_act):
    ntokens = len(corpus.dictionary)
    hidden = model.initVars(cuda, bsz)
    latent = model.initVars(cuda, bsz)
    content = model.initVars(cuda, bsz)

    input = autograd.Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)

    if cuda:
        input.data = input.data.cuda()

    generated = []
    for i in range(words):
        if tanh_act:
            content, latent, hidden, output = model(input, hidden, latent, content)
        else:
            latent, hidden, output = model(input, hidden, latent)
        word_weights = output.squeeze().data.div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.ix_to_word[word_idx]

        generated.append(word + ('\n' if i % 20 == 19 else ' '))

        if i % log_interval == 0:
            print('| Generated {}/{} words'.format(i, words))
    print(''.join(generated))


if __name__ == "__main__":
    BSZ = 20
    CUDA = True
    DATA_FILE = "../../data/penn/"
    TANH_ACT = False
    BPTT = 35
    model = torch.load("NNs/ppl41/test.pt")
    loss_function = nn.NLLLoss()
    corpus = Corpus(DATA_FILE, CUDA)
    vocab_size = len(corpus.dictionary)
    print("V", vocab_size)
    test_data = batchify(corpus.test, BSZ, CUDA)
    # print(math.exp(evaluate(model, corpus, loss_function, test_data, CUDA, BSZ, BPTT, TANH_ACT)))
    generate(corpus, CUDA, 1.0, 300, 1, 1, TANH_ACT)
