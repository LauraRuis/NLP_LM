import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from NLM import *
import json
from data import Corpus
from helpers import *
import time
import math
import random

torch.manual_seed(1)

# pars to change
CONTINUE_TRAINING = False #can either be True or False
CONTEXT_SIZE = 3
CUDA = False
DIRECT_CONNECTIONS = True
EMBEDDING_DIM = 50
BATCH_SIZE = 10
EPOCHS = 50
LR = 0.05
NN_FILENAME = "bengio.pt"
if DIRECT_CONNECTIONS:
    NN_FILENAME = "dc_bengio.pt"
DECODER = open("decoder.json", "w")
ENCODER = open("encoder.json", "w")
print_every = 5000
data_path = "../../data/penn/"

if CUDA:
    torch.cuda.manual_seed(1)

# read data
corpus = Corpus(data_path, CONTEXT_SIZE + 1)
training_data = batchify(corpus.train, BATCH_SIZE, CUDA, CONTEXT_SIZE)
val_data = batchify(corpus.valid, BATCH_SIZE, CUDA, CONTEXT_SIZE)

ntokens = len(corpus.dictionary)
print("|V|: ", ntokens)

losses = []
loss_function = nn.CrossEntropyLoss()
model = BengioNLM(ntokens, EMBEDDING_DIM, CONTEXT_SIZE, DIRECT_CONNECTIONS)

if CONTINUE_TRAINING:
    # Load the best saved model.
    with open(NN_FILENAME, 'rb') as f:
        model = torch.load(f, map_location=lambda storage, loc: storage)

    if CUDA:
        model.cuda()

optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-6)

word_to_ix = corpus.dictionary.word_to_ix
json.dump(corpus.dictionary.ix_to_word, DECODER, indent=4)
json.dump(word_to_ix, ENCODER, indent=4)

def train(current_epoch):

    total_loss = 0
    start_time = time.time()

    random.seed(current_epoch)
    random_indices = [batch for batch, i in enumerate(training_data)]
    random.shuffle(random_indices)
    # i = 0

    for i, k in enumerate(random_indices):
        # print("batch number: ", i)
        # print("random index: ", k)
        
        training_tuple = training_data[k]
        # print("training_tuple", training_tuple)

        training_tuple_0 = training_tuple[0].contiguous()
        context = autograd.Variable(training_tuple_0).view(BATCH_SIZE, CONTEXT_SIZE)
        target = autograd.Variable(training_tuple[1])

        model.zero_grad()
        log_probs = model(context)
        loss = loss_function(log_probs.view(-1, ntokens), target)
        loss.backward()

        optimizer.step()
        total_loss += loss.data

        if i % print_every == 0 and i > 0:
            cur_loss = total_loss[0] / print_every
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:6d}/{:6d} batches | lr {:02.3f} | ms/batch {:5.2f} | loss {:5.3f} | ppl {:8.2f}'
                  .format(epoch, i, len(training_data), LR,
                          elapsed * 1000 / print_every, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        # i += 1

best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(epoch)
        val_loss = evaluate(model, corpus, loss_function, val_data)
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

# Load the best saved model.
with open(NN_FILENAME, 'rb') as f:
    model = torch.load(f)