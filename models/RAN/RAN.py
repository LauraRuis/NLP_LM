import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RAN(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, tie_weights, softmax, dropout=0.5):
        super(RAN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.softmax = softmax
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.x2c = nn.Linear(embedding_size, hidden_size, bias=False)
        self.h2i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2i = nn.Linear(embedding_size, hidden_size)
        self.h2f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2f = nn.Linear(embedding_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)

        if tie_weights:
            if self.hidden_size != self.embedding_size:
                raise ValueError('When using the weight tying, hidden size must be equal to embedding size')
            self.h2o.weight = self.embeddings.weight

    def forward(self, word, hidden, latent, content=0):
        # forward method for simple RAN

        # set embeddings
        embeds = self.embeddings(word)
        temp_i = self.h2i(latent) + self.x2i(embeds)
        input_gate = F.sigmoid(temp_i)
        temp_f = self.h2f(latent) + self.x2f(embeds)
        forget_gate = F.sigmoid(temp_f)
        latent = torch.mul(input_gate, embeds) + torch.mul(forget_gate, latent)
        dropped = self.drop(latent)
        output = self.h2o(dropped)
        return latent, hidden, output

    def forward_dependencies(self, word, hidden, latent):
        # forward method that also returns input and forget gate

        # set embeddings
        embeds = self.embeddings(word)
        temp_i = self.h2i(latent) + self.x2i(embeds)
        input_gate = F.sigmoid(temp_i)
        temp_f = self.h2f(latent) + self.x2f(embeds)
        forget_gate = F.sigmoid(temp_f)
        latent = torch.mul(input_gate, embeds) + torch.mul(forget_gate, latent)
        dropped = self.drop(latent)
        output = self.h2o(dropped)
        return latent, hidden, output, input_gate, forget_gate

    def init_states(self, cud, bsz):
        hidden = Variable(torch.zeros(bsz, self.hidden_size))
        if cud:
            hidden = Variable(torch.zeros(bsz, self.hidden_size).cuda())
        return hidden
