import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RAN(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, tie_weights, softmax, nlayers, dropout=0.5):
        super(RAN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.softmax = softmax
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.nlayers = nlayers

        # for first layer
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.h2i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2i = nn.Linear(embedding_size, hidden_size)
        self.h2f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2f = nn.Linear(embedding_size, hidden_size)

        # for second layer
        self.h2i_2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2i_2 = nn.Linear(embedding_size, hidden_size)
        self.h2f_2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2f_2 = nn.Linear(embedding_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)

        if tie_weights:
            if self.hidden_size != self.embedding_size:
                raise ValueError('When using the weight tying, hidden size must be equal to embedding size')
            self.h2o.weight = self.embeddings.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.h2o.bias.data.fill_(0)
        self.h2o.weight.data.uniform_(-initrange, initrange)

    def forward(self, word, hidden, latent, content=0):

        # set embeddings
        embeds = self.embeddings(word)
        embeds = self.drop(embeds)

        # first layer
        # input gate
        temp_i = self.h2i(latent) + self.x2i(embeds)
        input_gate = F.sigmoid(temp_i)

        # forget gate
        temp_f = self.h2f(latent) + self.x2f(embeds)
        forget_gate = F.sigmoid(temp_f)

        # element wise multiplication
        latent = input_gate * embeds + forget_gate * latent

        # second layer
        # input gate
        temp_i_2 = self.h2i_2(latent) + self.x2i_2(embeds)
        input_gate_2 = F.sigmoid(temp_i_2)

        # forget gate
        temp_f_2 = self.h2f_2(latent) + self.x2f_2(embeds)
        forget_gate_2 = F.sigmoid(temp_f_2)

        # element wise multiplication
        latent_2 = input_gate_2 * embeds + forget_gate_2 * latent
        # output_2 = self.drop(latent_2)    
        output_2 = self.h2o(latent_2)

        return latent_2, hidden, output_2, input_gate_2, forget_gate_2

    def init_states(self, cud, bsz):
        weight = next(self.parameters()).data
        if cud:
            return Variable(weight.new(1, bsz, self.hidden_size).zero_().cuda())
        else:
            return Variable(weight.new(1, bsz, self.hidden_size).zero_())