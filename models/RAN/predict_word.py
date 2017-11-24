from RAN import *
from helpers import *
import json
import torch.autograd as autograd


def predict_word(model, context, encode, decode, cuda):

    context_idxs = [encode[w] for w in context]
    hidden = model.initVars(cuda)
    content = model.initVars(cuda)
    context_var = autograd.Variable(torch.cuda.LongTensor(context_idxs))
    latent, hidden, log_probs = model(context_var, hidden, content)
    max_idx = log_probs.view(-1).min(0)[1]
    idx = []
    for adim in list(log_probs.size())[::-1]:
        idx.append(max_idx % adim)
        max_idx = max_idx / adim
    idx = torch.cat(idx).data[0]
    print(idx)
    return decode[str(idx)]


def perplexity(model, data):
    print()


if __name__ == "__main__":

    data_file = "../../data/penn/test.txt"
    model = torch.load("test.pt")
    test_data = read_data(data_file)
    decoder_file = open("decoder.json", "r")
    encoder_file = open("encoder.json", "r")
    decoder = json.load(decoder_file)
    encoder = json.load(encoder_file)
    test_sent = "she"
    next_word = predict_word(model, test_sent.split(), encoder, decoder, True)
    print(test_sent + " " + next_word)
