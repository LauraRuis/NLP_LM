import torch.nn as nn
import torch.nn.functional as F


class BengioNLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, direct_connections=False):
        super(BengioNLM, self).__init__()
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 30)
        self.linear2 = nn.Linear(30, vocab_size)
        self.linear3 = nn.Linear(context_size * embedding_dim, vocab_size, bias=False)
        self.dc = direct_connections

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(-1, self.context_size * self.embedding_dim)
        out = F.tanh(self.linear1(embeds))
        y = self.linear2(out)
        if self.dc:
            y = y + self.linear3(embeds)
        # log_probs = F.log_softmax(y)
        return y
