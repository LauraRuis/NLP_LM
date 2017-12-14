import torch.optim as optim
from RAN import *
import json
from helpers import *
from data import Corpus
import time
import torch
import math
import random

# set seed
torch.manual_seed(1)

########################################################################################################################
# adjustable parameters
########################################################################################################################
TIE_WEIGHTS = True
DROPOUT = 0
EMBEDDING_DIM = 650
BPTT = 35
BSZ = 20
EVAL_BSZ = 10
HIDDEN_SIZE = 650
LR = 1.5
CLIP = False
LAYERS = 2
########################################################################################################################
EPOCHS = 150
PRINT_EVERY = 500
CUDA = True
LOSS_FUNC = "CrossEnt"  # options: CrossEnt or NLLLoss (if using NLLLoss add softmax to forward method before training)
DATA_FILE = "../../data/penn/"  # options: Penn Treebank or Alphabet
CONTINUE_TRAINING = True  # use if want to continue training on old pt file
########################################################################################################################

# save decoders
DECODER = open("decoder.json", "w")
ENCODER = open("encoder.json", "w")

# set filename for saving parameters
NN_FILENAME = \
    "NNs/hidden_" + str(HIDDEN_SIZE) + \
    "-embed_" + str(EMBEDDING_DIM) + \
    "_drop_" + str(DROPOUT) + \
    "_layers_" + str(LAYERS) + \
    "_tying_" + str(TIE_WEIGHTS) + ".pt"

# read data
corpus = Corpus(DATA_FILE, CUDA)
vocab_size = len(corpus.dictionary)
if CONTINUE_TRAINING:
    SAVED_NN = NN_FILENAME
    print("Continuing training on old model in file ", NN_FILENAME)
print("|V|", vocab_size)

# turn into batches
training_data = batchify(corpus.train, BSZ, CUDA)
validation_data = batchify(corpus.valid, EVAL_BSZ, CUDA)

# set loss function
if LOSS_FUNC == "CrossEnt":
    loss_function = nn.CrossEntropyLoss()
    softmax = False
elif LOSS_FUNC == "NLLLoss":
    loss_function = nn.NLLLoss()
    softmax = True
else:
    loss_function = nn.CrossEntropyLoss()
    softmax = False

# Load the best saved model or initialize new one
if CONTINUE_TRAINING:
    with open(SAVED_NN, 'rb') as f:
        model = torch.load(f)
else:
    # initialize model
    model = RAN(EMBEDDING_DIM, vocab_size, HIDDEN_SIZE, TIE_WEIGHTS, softmax, LAYERS, DROPOUT)

if CUDA:
    model.cuda()
    torch.cuda.manual_seed(1)

# save encoder en decoder to json file
json.dump(corpus.dictionary.ix_to_word, DECODER, indent=4)
json.dump(corpus.dictionary.word_to_ix, ENCODER, indent=4)


def train(current_epoch):

    # enable dropout
    model.train()

    # initialize loss per epoch
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    # initialize hidden, latent and content layers with zero filled tensors
    hidden = model.init_states(CUDA, BSZ)
    latent = model.init_states(CUDA, BSZ)

    # shuffle indices to loop through data in random order
    random_indices = [i for batch, i in enumerate(range(0, training_data.size(0) - 1, BPTT))]
    random.seed(current_epoch)
    random.shuffle(random_indices)

    # loop over all data
    for batch, i in enumerate(range(0, training_data.size(0) - 1, BPTT)):

        # get batch of training data
        context, target = get_batch(training_data, random_indices[batch], BPTT)

        ################################################################################################################
        # code for testing if sequence still correct
        #
        # print([corpus.dictionary.ix_to_word[idx] for idx in target.data])
        # for i, batch in enumerate(context.data):
        #     print([corpus.dictionary.ix_to_word[idx] for idx in batch])
        #     print([corpus.dictionary.ix_to_word[target.data[i * BSZ + j]] for j in range(BPTT)])
        #     print([j * BPTT + i for j in range(BSZ)])
        # print([corpus.dictionary.ix_to_word[idx] for idx in context.data])
        # print([corpus.dictionary.ix_to_word[idx] for idx in target.data])
        ################################################################################################################

        # repackage hidden stop backprop from going to beginning each time
        hidden = repackage_hidden(hidden)
        latent = repackage_hidden(latent)

        # set gradients to zero
        model.zero_grad()

        # only use batch if dimensions are correct
        if target.size(0) == BSZ * BPTT:

            # forward pass
            latent, log_probs, _, _ = model(context, latent)

            # get the loss
            loss = loss_function(log_probs.view(-1, ntokens), target)

            # back propagate
            loss.backward()

            # clip gradients to get rid of exploding gradients problem
            if CLIP > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)

            # update parameters
            optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-6)
            optimizer.step()
            # for p in model.parameters():
            #     p.data.add_(-LR, p.grad.data)

            # update total loss
            total_loss += loss.data

            # print progress
            if batch % PRINT_EVERY == 0 and batch > 0:
                cur_loss = total_loss[0] / PRINT_EVERY
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(training_data) // BPTT, LR,
                                  elapsed * 1000 / PRINT_EVERY, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()


# initialize best validation loss
best_val_loss = None

# hit Ctrl + C to break out of training early
try:

    # loop over epochs
    for epoch in range(1, EPOCHS):

        epoch_start_time = time.time()
        train(epoch)
        val_loss = evaluate(model, corpus, loss_function, validation_data, CUDA, EVAL_BSZ, BPTT)
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

        # Anneal the learning rate if no improvement has been seen in the validation data set.
        if epoch > 6:
            LR /= 1.2

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
