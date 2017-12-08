import torch.optim as optim
from RAN import *
import json
from helpers import *
from data import Corpus
import time
import torch
import math

# set seed
torch.manual_seed(1)

########################################################################################################################
# adjustable parameters
########################################################################################################################
TANH_ACT = False
TIE_WEIGHTS = True
CUDA = True
DROPOUT = 0.5
EMBEDDING_DIM = 50
BPTT = 35
BSZ = 10
HIDDEN_SIZE = 50
EPOCHS = 50
PRINT_EVERY = 500
LR = 20
CLIP = 0.25
LOSS_FUNC = "CrossEnt"  # options: CrossEnt or NLLLoss
DATA_FILE = "../../data/penn/"
CONTINUE_TRAINING = False
SAVED_NN = "hidden_50-embed_50_tanh_False_drop_0.5_tying_True.pt"
########################################################################################################################

# save decoders
DECODER = open("decoder.json", "w")
ENCODER = open("encoder.json", "w")

# set filename for saving parameters
NN_FILENAME = \
    "NNs/hidden_" + str(HIDDEN_SIZE) + \
    "-embed_" + str(EMBEDDING_DIM) + \
    "_tanh_" + str(TANH_ACT) + \
    "_drop_" + str(DROPOUT) + \
    "_tying_" + str(TIE_WEIGHTS) + "/model.pt"

# read data
corpus = Corpus(DATA_FILE, CUDA)
vocab_size = len(corpus.dictionary)
if CONTINUE_TRAINING:
    print("Continuing training on old model in file ", NN_FILENAME)
print("|V|", vocab_size)

# turn into batches
training_data = batchify(corpus.train, BSZ, CUDA)
validation_data = batchify(corpus.valid, BSZ, CUDA)

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
    model = RAN(EMBEDDING_DIM, vocab_size, HIDDEN_SIZE, TIE_WEIGHTS, softmax, DROPOUT, TANH_ACT)

if CUDA:
    model.cuda()
    torch.cuda.manual_seed(1)

optimizer = optim.SGD(model.parameters(), lr=LR)

# save encoder en decoder to json file
json.dump(corpus.dictionary.ix_to_word, DECODER, indent=4)
json.dump(corpus.dictionary.word_to_ix, ENCODER, indent=4)


def train():

    # enable dropout
    model.train()

    # initialize loss per epoch
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    # initialize hidden, latent and content layers with zero filled tensors
    hidden = model.initVars(CUDA, BSZ)
    latent = model.initVars(CUDA, BSZ)
    content = model.initVars(CUDA, BSZ)

    # loop over all data
    for batch, i in enumerate(range(0, training_data.size(0) - 1, BPTT)):

        # get batch of training data
        context, target = get_batch(training_data, i, BPTT)

        # repackage hidden stop backprop from going to beginning each time
        hidden = repackage_hidden(hidden)

        # set gradients to zero
        model.zero_grad()

        # only use batch if dimensions are correct
        if target.size(0) == BSZ * BPTT:

            # depending on choice add tanh nonlinearity or not
            if TANH_ACT:
                content, latent, hidden, log_probs = model(context, hidden, latent, content)
            else:
                latent, hidden, log_probs = model(context, hidden, latent)

            # get the loss
            loss = loss_function(log_probs.view(-1, ntokens), target)

            # back propagate
            if batch < len(training_data) // BPTT - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            # clip gradients to get rid of exploding gradients problem
            torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)

            # update parameters
            optimizer.step()

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
