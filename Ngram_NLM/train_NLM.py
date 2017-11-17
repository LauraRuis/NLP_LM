import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from NLM import *
import json
from helpers import *

torch.manual_seed(1)

# pars to change
CONTEXT_SIZE = 2
EMBEDDING_DIM = 5
EPOCHS = 10
NN_FILENAME = "nlm.pt"
DECODER = open("decoder.json", "w")
ENCODER = open("encoder.json", "w")
print_every = 1
data_file = "../data/penn/train.txt"

# read data
training_data = read_data(data_file)

# build a list of tuples for trigrams
trigrams = n_grams(3, training_data)

vocab = set(training_data)
print("V: ", len(vocab))

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

json.dump(ix_to_word, DECODER, indent=4)
json.dump(word_to_ix, ENCODER, indent=4)

for epoch in range(EPOCHS):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        # turn the words into integer indices and wrap them in variables
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        model.zero_grad()
        log_probs = model(context_var)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        loss.backward()
        optimizer.step()

        total_loss += loss.data
        
    if epoch % print_every == 0:
        print("epoch: ", epoch, " of ", EPOCHS)
        print("Current loss")	
        print(loss.data[0])

    losses.append(total_loss)

print(losses)
torch.save(model, NN_FILENAME)

