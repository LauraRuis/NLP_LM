import torch.autograd as autograd
import torch


def read_data(fn):

    data = ""
    f = open(fn, "r")
    
    for word in f.read().split():
        data += word + " "

    return data.split()


def read_sentences(fn):

    sentences = []
    data = ""
    f = open(fn, "r")

    for sentence in f.read().split("\n"):
        sentences.append(sentence.split())
        for word in sentence.split():
            data += word + " "

    return data.split(), sentences


def n_grams(n, data):
    
    context_size = n - 1
    ngrams = []
    for i in range(len(data) - context_size):
        context = []
        for j in range(n - 1):
            context.append(data[i + j])
        target = data[i + n - 1]
        ngrams.append((context, target))

    return ngrams


def batchify(data, bsz, cuda):

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)

    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()

    if cuda:
        data.cuda()

    return data


def get_batch(source, i, bptt, evaluation=False):
    seq_len = min(bptt, len(source) - 1 - i)
    data = autograd.Variable(source[i:i + seq_len], volatile=evaluation)
    target = autograd.Variable(source[i + 1:i + 1 + seq_len].view(-1))

    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == autograd.Variable:
        return autograd.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(model, corpus, criterion, data_source, cuda, bsz, bptt, tanh_act):
    # Turn on evaluation mode which disables dropout.
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.initVars(cuda, bsz)
    latent = model.initVars(cuda, bsz)
    content = model.initVars(cuda, bsz)
    for i in range(0, data_source.size(0) - 1, bptt):
        context, target = get_batch(data_source, i, bptt, evaluation=True)
        if target.size(0) == bsz * bptt:
            if tanh_act:
                content, latent, hidden, log_probs = model(context, hidden, latent, content)
            else:
                latent, hidden, log_probs = model(context, hidden, latent)
            output_flat = log_probs.view(-1, ntokens)
            total_loss += len(context) * criterion(output_flat, target).data
            hidden = repackage_hidden(hidden)

    return total_loss[0] / len(data_source)
