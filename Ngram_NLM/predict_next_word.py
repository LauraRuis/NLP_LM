import torch
from NLM import *
import sys
import torch.nn as nn
from helpers import *
import json
import torch.autograd as autograd

data_file = "../data/penn/test.txt"
model = torch.load("nlm.pt")
test_data = read_data(data_file)[:10000]
decoder_file = open("decoder.json", "r")
encoder_file = open("encoder.json", "r")
decoder = json.load(decoder_file)
encoder = json.load(encoder_file)
test_sent = "this is".split()
context_idxs = [encoder[w] for w in test_sent]
context_var = autograd.Variable(torch.LongTensor(context_idxs))
logProbs = model(context_var)
rawmaxidx = logProbs.view(-1).min(0)[1]
idx = []
for adim in list(logProbs.size())[::-1]:
    idx.append(rawmaxidx%adim)
    rawmaxidx = rawmaxidx / adim
idx = torch.cat(idx).data[0]
print(decoder[str(idx)])
