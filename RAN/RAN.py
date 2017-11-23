import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RAN(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, act_function=True, pretrained=True):
        super(RAN, self).__init__()
        self.pretrained = pretrained
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.activation_func = act_function
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.x2c = nn.Linear(embedding_size, hidden_size, bias=False)
        self.h2i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2i = nn.Linear(embedding_size, hidden_size)
        self.h2f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2f = nn.Linear(embedding_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)

    def forward(self, word, hidden, latent, content=0):

        # use pre-trained embeddings or make own embeddings
        if not self.pretrained:
            embeds = self.embeddings(word).view((1, -1))
        else:
            embeds = []

        # nonlinear activation function or identity activation function
        if self.activation_func:
            content = self.x2c(embeds)
            temp_i = self.h2i(hidden) + self.x2i(embeds)
            input_gate = F.sigmoid(temp_i)
            temp_f = self.h2f(hidden) + self.x2f(embeds)
            forget_gate = F.sigmoid(temp_f)
            latent = torch.mul(input_gate, content) + torch.mul(forget_gate, latent)
            hidden = F.tanh(latent)
            output = self.h2o(hidden)
            output = F.log_softmax(output)
            return content, latent, hidden, output
        else:
            temp_i = self.h2i(latent) + self.x2i(embeds)
            input_gate = F.sigmoid(temp_i)
            temp_f = self.h2f(latent) + self.x2f(embeds)
            forget_gate = F.sigmoid(temp_f)
            latent = torch.mul(input_gate, embeds) + torch.mul(forget_gate, latent)
            output = self.h2o(latent)
            output = F.log_softmax(output)
            return latent, hidden, output

    def initVars(self, cud):
        hidden = Variable(torch.zeros(1, self.hidden_size))
        if cud:
            hidden = Variable(torch.zeros(1, self.hidden_size).cuda())
        return hidden
