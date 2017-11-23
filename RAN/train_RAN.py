import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from RAN import *
import json
from random import shuffle
from helpers import *

torch.manual_seed(1)

# pars to change
TANH_ACT = False
PRETRAINED = False
CUDA = True
EMBEDDING_DIM = 5
HIDDEN_SIZE = 5
EPOCHS = 1
NN_FILENAME = "hidden_" + str(HIDDEN_SIZE) + "-embed_" + str(EMBEDDING_DIM) + "tanh_" + str(TANH_ACT) + ".pt"
DECODER = open("decoder.json", "w")
ENCODER = open("encoder.json", "w")
PRINT_EVERY = 1
LOSS_EVERY = 1
DATA_FILE = "../data/penn/train.txt"

# read data
words, sentences = read_sentences(DATA_FILE)
vocab = set(words)
vocab_size = len(vocab)
num_sentence = len(sentences)
print("V", vocab_size)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}

losses = []
loss_function = nn.NLLLoss()
model = RAN(EMBEDDING_DIM, vocab_size, HIDDEN_SIZE, TANH_ACT, PRETRAINED)
if CUDA:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001)

json.dump(ix_to_word, DECODER, indent=4)
json.dump(word_to_ix, ENCODER, indent=4)

average_loss = 0
all_losses = []
for epoch in range(EPOCHS):

    # initialize loss per epoch
    total_loss = torch.Tensor([0])
    if CUDA:
        total_loss = total_loss.cuda()

    # randomize sentence order
    shuffle(sentences)

    # loop over all data
    count = 0
    for sentence in sentences:
        print("Sentence: ", count, " of ", num_sentence)
        count += 1
        model.zero_grad()

        hidden = model.initVars(CUDA)
        latent = model.initVars(CUDA)
        content = model.initVars(CUDA)

        # loop over sentence
        for i, word in enumerate(sentence[:-1]):

            # turn the words into integer indices and wrap them in variables
            context_idx = word_to_ix[word]
            context = torch.LongTensor([context_idx])
            target_word = sentence[i + 1]
            target = torch.LongTensor([word_to_ix[target_word]])

            if CUDA:
                context = context.cuda()
                target = target.cuda()

            context_var = autograd.Variable(context)
            target_var = autograd.Variable(context)

            if TANH_ACT:
                content, latent, hidden, log_probs = model(context_var, hidden, latent, content)
            else:
                latent, hidden, log_probs = model(context_var, hidden, latent)

            loss = loss_function(log_probs, target_var)
            if i < len(sentence) - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizer.step()

            total_loss += loss.data
            average_loss += loss.data

    if epoch % LOSS_EVERY == 0:
        avg_loss = average_loss.data[0] / LOSS_EVERY
        print("Total loss: ", avg_loss)
        all_losses.append(avg_loss)

    if epoch % PRINT_EVERY == 0:
        print("epoch: ", epoch, " of ", EPOCHS)
        print("Current loss")
        print(loss.data[0])

    losses.append(total_loss)

print(losses)
torch.save(model, NN_FILENAME)
torch.save(model, "test.pt")
