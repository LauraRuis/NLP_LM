import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from RAN import *
import json
from helpers import *
from data import Corpus
import time
import math

# torch.cuda.device(0)

torch.manual_seed(0)

# pars to change
TANH_ACT = False
TIE_WEIGHTS = True
PRETRAINED = False
CUDA = True
DROPOUT = 0.5
EMBEDDING_DIM = 50
BPTT = 35
BSZ = 10
HIDDEN_SIZE = 50
EPOCHS = 50
NN_FILENAME = \
    "hidden_" + str(HIDDEN_SIZE) + \
    "-embed_" + str(EMBEDDING_DIM) + \
    "_tanh_" + str(TANH_ACT) + \
    "_drop_" + str(DROPOUT) + \
    "_tying_" + str(TIE_WEIGHTS) + ".pt"

DECODER = open("decoder.json", "w")
ENCODER = open("encoder.json", "w")
PRINT_EVERY = 100
LR = 20
DATA_FILE = "../../data/penn/"


# read data
corpus = Corpus(DATA_FILE, CUDA)
vocab_size = len(corpus.dictionary)
print("V", vocab_size)

training_data = batchify(corpus.train, BSZ, CUDA)
validation_data = batchify(corpus.valid, BSZ, CUDA)

losses = []
loss_function = nn.CrossEntropyLoss()
model = RAN(EMBEDDING_DIM, vocab_size, HIDDEN_SIZE, TIE_WEIGHTS, DROPOUT, TANH_ACT, PRETRAINED)

if CUDA:
    model.cuda()
    torch.cuda.manual_seed(0)

# Load the best saved model.
# with open(NN_FILENAME, 'rb') as f:
#     model = torch.load(f)

optimizer = optim.SGD(model.parameters(), lr=LR)

json.dump(corpus.dictionary.ix_to_word, DECODER, indent=4)
json.dump(corpus.dictionary.word_to_ix, ENCODER, indent=4)

average_loss = 0
all_losses = []


def train():

    # initialize loss per epoch
    total_loss = 0

    start_time = time.time()
    ntokens = len(corpus.dictionary)

    # init vars
    hidden = model.initVars(CUDA, BSZ)
    latent = model.initVars(CUDA, BSZ)
    content = model.initVars(CUDA, BSZ)

    # loop over all data

    for batch, i in enumerate(range(0, training_data.size(0) - 1, BPTT)):

        context, target = get_batch(training_data, i, BPTT)

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        if target.size(0) == BSZ * BPTT:

            if TANH_ACT:
                content, latent, hidden, log_probs = model(context, hidden, latent, content)
            else:
                latent, hidden, log_probs = model(context, hidden, latent)

            loss = loss_function(log_probs.view(-1, ntokens), target)

            if batch < len(training_data) // BPTT - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            optimizer.step()

            total_loss += loss.data
            if batch % PRINT_EVERY == 0 and batch > 0:
                cur_loss = total_loss[0] / PRINT_EVERY
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(training_data) // BPTT, LR,
                                  elapsed * 1000 / PRINT_EVERY, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    losses.append(total_loss)


# Loop over epochs.
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, EPOCHS):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, corpus, loss_function, validation_data, CUDA, BSZ, BPTT, TANH_ACT)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(NN_FILENAME, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            LR /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

with open(NN_FILENAME, 'wb') as f:
    torch.save(model, f)
