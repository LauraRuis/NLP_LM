import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from NLM import *
import json
from helpers import *

torch.manual_seed(1)

# pars to change
CONTEXT_SIZE = 2
CUDA = True
EMBEDDING_DIM = 50
EPOCHS = 10
NN_FILENAME = "bengio.pt"
DECODER = open("decoder.json", "w")
ENCODER = open("encoder.json", "w")
print_every = 1
data_file = "../../data/penn/train.txt"

# read data
training_data = read_data(data_file)

# build a list of tuples for trigrams
ngrams = n_grams(CONTEXT_SIZE + 1, training_data)

vocab = set(training_data)
print("V: ", len(vocab))

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

losses = []
loss_function = nn.NLLLoss()
model = BengioNLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
if CUDA:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001)

json.dump(ix_to_word, DECODER, indent=4)
json.dump(word_to_ix, ENCODER, indent=4)

for epoch in range(EPOCHS):
    total_loss = torch.Tensor([0])
    if CUDA:
        total_loss = torch.Tensor([0]).cuda()
    for context, target in ngrams:
        # turn the words into integer indices and wrap them in variables
        context_idxs = [word_to_ix[w] for w in context]

        context_tensor = torch.LongTensor(context_idxs)
        if CUDA:
            context_tensor = torch.cuda.LongTensor(context_idxs)
        context_var = autograd.Variable(context_tensor)

        model.zero_grad()
        log_probs = model(context_var)
        
        target_tensor = torch.LongTensor([word_to_ix[target]])
        if CUDA:
            target_tensor = torch.cuda.LongTensor([word_to_ix[target]])
        target_var = autograd.Variable(target_tensor)
        loss = loss_function(log_probs, target_var)

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

