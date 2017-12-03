import torch.autograd as autograd
import torch


def read_data(fn):

    data = ""
    f = open(fn, "r")
    
    for word in f.read().split():
        data += word + " "

    return data.split()


def batchify(data, bsz, cuda, context_size):

    batches = []
    for i in range(0, len(data), bsz):
        ngrams = data[i:i + bsz]
        context_batch = []
        target_batch = []
        for context, target in ngrams:
            context_batch.append(context.view(context_size, 1))
            target_batch.append(target.view(-1))
        context_batch = torch.cat(context_batch, 1).t()
        target_batch = torch.cat(target_batch, 0)
        if cuda:
            context_batch = context_batch.cuda()
            target_batch = target_batch.cuda()
        if context_batch.size()[0] == bsz:
            batches.append((context_batch, target_batch))

    return batches


def evaluate(model, corpus, criterion, data_source):
    # Turn on evaluation mode which disables dropout.
    total_loss = 0
    ntokens = len(corpus.dictionary)
    for batch, training_tuple in enumerate(data_source):
        context = autograd.Variable(training_tuple[0])
        target = autograd.Variable(training_tuple[1])
        log_probs = model(context)
        log_probs_flat = log_probs.view(-1, ntokens)
        total_loss += criterion(log_probs_flat, target).data
    return total_loss[0] / len(data_source)
