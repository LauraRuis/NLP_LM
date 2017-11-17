def read_data(fn):

    data = ""
    f = open(fn, "r")
    
    for word in f.read().split():
        data += word + " "

    return data.split()


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
