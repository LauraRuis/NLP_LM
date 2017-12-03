import torch
from NLM import *
from helpers import *
import json
import torch.autograd as autograd
from helpers import *
import numpy as np
from scipy.misc import logsumexp
import math
from data import Corpus

CUDA = True
CONTEXT_SIZE = 3


def predict_word(model, context, encode, decode):

    context_idxs = [encode[w] for w in context]
    context_var = autograd.Variable(torch.cuda.LongTensor(context_idxs))
    log_probs = model(context_var)
    max_idx = log_probs.view(-1).min(0)[1]
    idx = []
    for adim in list(log_probs.size())[::-1]:
        idx.append(max_idx % adim)
        max_idx = max_idx / adim
    idx = torch.cat(idx).data[0]

    return decode[str(idx)]


def perplexity(model):

    loss_function = nn.NLLLoss()

    data_path = "../../data/penn/"

    # read data
    corpus = Corpus(data_path, CONTEXT_SIZE + 1)
    test_data = batchify(corpus.test, 10, CUDA, CONTEXT_SIZE)
    test_loss = evaluate(model, corpus, loss_function, test_data)
    print("perplexity as exp(average_loss): ", math.exp(test_loss))

    return math.exp(test_loss)


if __name__ == "__main__":

    model = torch.load("dc_bengio_temp.pt")
    decoder_file = open("decoder.json", "r")
    encoder_file = open("encoder.json", "r")
    decoder = json.load(decoder_file)
    encoder = json.load(encoder_file)
    # test_sent = "he is	 "
    # next_word = predict_word(model, test_sent.split(), encoder, decoder)
    # print(test_sent + next_word)
    perplex = perplexity(model)
