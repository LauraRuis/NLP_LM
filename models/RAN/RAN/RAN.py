import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RAN(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, tie_weights, softmax, nlayers, dropout=0.5):
        super(RAN, self).__init__()
        self.drop = nn.Dropout(dropout, inplace=False)
        self.hidden_size = hidden_size
        self.softmax = softmax
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.nlayers = nlayers

        # input and output layer weights
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)

        # weights for first layer
        self.h2i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2i = nn.Linear(embedding_size, hidden_size)
        self.h2f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2f = nn.Linear(embedding_size, hidden_size)
        self.weights_layer_one = self.h2i, self.x2i, self.h2f, self.x2f

        # weights for second layer
        self.h2i_2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2i_2 = nn.Linear(embedding_size, hidden_size)
        self.h2f_2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2f_2 = nn.Linear(embedding_size, hidden_size)
        self.weights_layer_two = self.h2i_2, self.x2i_2, self.h2f_2, self.x2f_2

        if tie_weights:
            if self.hidden_size != self.embedding_size:
                raise ValueError('When using the weight tying, hidden size must be equal to embedding size')
            self.h2o.weight = self.embeddings.weight

        self.init_weights(0.1)

    def init_weights(self, initrange):

        # input and output layer weights
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.h2o.bias.data.fill_(0)
        self.h2o.weight.data.uniform_(-initrange, initrange)

        # first layer weights
        self.h2i.weight.data.uniform_(-initrange, initrange)
        self.x2i.weight.data.uniform_(-initrange, initrange)
        self.x2i.bias.data.fill_(0)
        self.h2f.weight.data.uniform_(-initrange, initrange)
        self.x2f.weight.data.uniform_(-initrange, initrange)
        self.x2f.bias.data.fill_(0)

        # second layer weights
        self.h2i_2.weight.data.uniform_(-initrange, initrange)
        self.x2i_2.weight.data.uniform_(-initrange, initrange)
        self.x2i_2.bias.data.fill_(0)
        self.h2f_2.weight.data.uniform_(-initrange, initrange)
        self.x2f_2.weight.data.uniform_(-initrange, initrange)
        self.x2f_2.bias.data.fill_(0)

    def forward(self, word, latent, content=0):

        # set embeddings and dropout
        embeds = self.embeddings(word)
        embeds = self.drop(embeds)

        # first layer
        latents = []
        for i in range(embeds.size()[0]):
            latent, i_gate, f_gate = self.RAN(embeds[i], latent, self.weights_layer_one)
            latents.append(latent)

        inputs = torch.cat(latents, 0).view(embeds.size(0), *latents[0].size())
        # inputs = self.drop(inputs)

        # second layer
        outputs = []
        for i in range(embeds.size()[0]):
            latent, i_gate, f_gate = self.RAN(inputs[i], latent, self.weights_layer_two)
            outputs.append(latent)

        outputs = torch.cat(outputs, 0).view(embeds.size(0), *outputs[0].size())

        # dropout before going through output layer
        outputs = self.drop(outputs)
        outputs = self.h2o(outputs)

        return latent, outputs, i_gate, f_gate

    def RAN(self, embeds, latent, weights):

        # get weights
        self.h2i, self.x2i, self.h2f, self.x2f = weights

        # input gate
        input_gate = F.sigmoid(self.h2i(latent) + self.x2i(embeds))

        # forget gate
        forget_gate = F.sigmoid(self.h2f(latent) + self.x2f(embeds))

        # element wise multiplication
        latent = input_gate * embeds + forget_gate * latent

        return latent, input_gate, forget_gate

    def init_states(self, cud, bsz):
        weight = next(self.parameters()).data
        if cud:
            return Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda())
        else:
            return Variable(weight.new(1, bsz, self.hidden_size).zero_())
