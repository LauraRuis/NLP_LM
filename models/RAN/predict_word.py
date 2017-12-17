import torch.nn as nn
import torch
from helpers import *
import torch.autograd as autograd
from data import Corpus
import math
import numpy as np
import displacy
import argparse
from collections import defaultdict
import pickle

BSZ = 20
BPTT = 35


def predict_word(corpus, model, context, cuda):

    # disable dropout
    model.eval()

    word_to_ix = corpus.dictionary.word_to_ix
    context = torch.cuda.LongTensor([word_to_ix[context]])
    latent = model.init_states(cuda, 1)
    context = autograd.Variable(context)
    latent, log_probs, _, _ = model(context, latent)
    max = log_probs.view(-1).max()
    idx = 0
    for i, el in enumerate(log_probs.view(-1)):
        if el.data[0] == max.data[0]:
            idx = i
    return corpus.dictionary.ix_to_word[idx]


def generate(corpus, cuda, temperature, words, log_interval, bsz):
    ntokens = len(corpus.dictionary)
    latent = model.init_states(cuda, bsz)

    input = autograd.Variable(torch.rand(
        1, 1).mul(ntokens).long(), volatile=True)

    if cuda:
        input.data = input.data.cuda()

    generated = []
    for i in range(words):
        latent, output, _, _ = model(input, latent)
        word_weights = output.squeeze().data.div(temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.ix_to_word[word_idx]

        generated.append(word + ('\n' if i % 20 == 19 else ' '))

        if i % log_interval == 0:
            print('| Generated {}/{} words'.format(i, words))
    print(''.join(generated))


def prepareModelAnalysys(model, corpus, sentence, cuda):
    hidden = model.init_states(cuda, 1)  # bsz = 1
    latent = model.init_states(cuda, 1)  # bsz = 1

    model.eval()

    context_tensor = torch.LongTensor(
        [corpus.dictionary.word_to_ix[word] for word in sentence])

    if cuda:
        context_tensor.cuda()

    context = autograd.Variable(context_tensor)

    hidden, latent, log_probs, i_gates, f_gates = model(
        context, hidden, latent, weight_analysis=True)

    input_w_layers = []
    forget_w_layers = []
    for layer in range(model.nlayers):
        input_gates = []
        forget_gates = []
        for weight in i_gates[layer]:
            input_gates.append(weight.data.numpy())
        for weight in f_gates[layer]:
            forget_gates.append(weight.data.numpy())
        input_w_layers.append(input_gates)
        forget_w_layers.append(forget_gates)
    return input_w_layers, forget_w_layers


def visualize(display, words, arcs, filename):
    with open(filename, 'w') as f:
        # arcs = []
        # for i, arc in enumerate(arcs):
        #     # if i != index:
        #     arcs.append((i+1, index))
        f.write(display.render(displacy.parseWordsArcs(words, arcs)))


def analysisArcLength(datafile, model_name, model, corpus, norm, cuda, limit_sentences):
    print("***********"+datafile+'****************')
    layers_maxarc = []
    layers_avgarc = []
    layer_max_lengths_med = []
    layer_med_lengths_med = []
    for i in range(model.nlayers):
        layers_maxarc.append(defaultdict(list))
        layers_avgarc.append(defaultdict(list))
        layer_max_lengths_med.append(defaultdict(float))
        layer_med_lengths_med.append(defaultdict(float))

    with open(datafile+'/test.txt', 'r') as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            words = stripped.split(' ')
            # print(words)
            if len(words) < 2:
                continue
            if i % 100 == 0:
                print('sentences processed:', i)
            if i > limit_sentences:
                break
            arc_lengths_layer = word_dependencies(model, corpus, words, norm,
                                                  norm+'_'+datafile[-5:-1]+'_'+model_name+'.svg', cuda, False)
            # print(len(words), arc_lengths_layer)
            for layer_id in range(model.nlayers):

                layers_maxarc[layer_id][str(len(words))].append(
                    max(arc_lengths_layer[layer_id]))

                layers_avgarc[layer_id][str(len(words))].append(
                    np.average(arc_lengths_layer[layer_id]))
    # print(layers_avgarc)
    for i in range(model.nlayers):
        for key, items in layer_max_lengths_med[i].items():
            layer_max_lengths_med[i][key] = np.median(items)

        for key, items in layer_med_lengths_med[i].items():
            layer_med_lengths_med[i][key] = np.median(items)

    return layer_max_lengths_med, layer_med_lengths_med, layers_maxarc, layers_avgarc


def word_dependencies(model, corpus, sentence, norm, filename, cuda=None,
                      viz=False, colors=['#000000', '#f4425f']):

    input_w_layers, forget_w_layers = prepareModelAnalysys(
        model, corpus, sentence, cuda)
    arcs_per_layers = []

    display = displacy.Displacy({'wordDistance': 130, 'arcDistance': 40,
                                 'wordSpacing': 30, 'arrowSpacing': 10})

    # loop over words
    for layer in range(model.nlayers):
        arcs = []
        for i, word in enumerate(sentence):

            # if word has a history
            if i > 0:

                # list to save weights of words in history
                weights = []
                # loop over its history
                for j in range(i):

                    # multiply all subsequent forget gates
                    forget_gates = forget_w_layers[layer][j + 1]
                    for h in range(j + 2, i + 1):
                        forget_gates = np.multiply(forget_gates, forget_w_layers[
                            layer][h])

                    weight = np.multiply(forget_gates, input_w_layers[
                        layer][j])

                    if norm == "max":
                        weights.append(np.max(weight))
                    elif norm == "sum":
                        weights.append(np.sum(weight))
                    elif norm == "l2":
                        weights.append(np.sum(np.power(weight, 2)))

                # if norm == "l2":
                    # weights = [w for w in weights]
                for k in range(len(colors)):
                    index = weights.index(np.max(weights))
                    weights[index] = 0
                    arcs.append((i, index, colors[k]))
                    if i <= len(colors)-1:
                        break
        arcs_per_layers.append(arcs)
        # print('layer ' + str(layer) + ': ', important_idxs)
        if viz:
            visualize(display, sentence, arcs, str(layer)+filename)
    # print(arcs_per_layers)
    arc_lengths_layer = []
    for layer_arcs in arcs_per_layers:
        layer = []
        for arc in layer_arcs:
            if arc[2] == colors[0]:
                layer.append(arc[0]-arc[1])
        arc_lengths_layer.append(layer)
    return arc_lengths_layer


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict RAN.')
    parser.add_argument('-d', '--datafile', default='../../data/penn/',
                        help='Dataset file')
    parser.add_argument('-m', '--model', default='NNs/hidden_650-embed_650_drop_0.5_layers_1_tying_True.pt',
                        help='Model file')
    parser.add_argument('-n', '--name', default='one_layer',
                        help='Model name for output file naming')
    parser.add_argument('--norm', default='max',
                        help='[max,sum,l2]')
    parser.add_argument('--perplexity', action='store_true',
                        help='Calculate perplexity')

    args = parser.parse_args()

    ##########################################################################
    # parameters to change
    ##########################################################################
    BSZ = 20
    CUDA = False
    DATA_FILE = args.datafile
    TANH_ACT = False
    BPTT = 35
    model_working_weight_analysis = args.model
    model = torch.load(model_working_weight_analysis,
                       map_location=lambda storage, loc: storage)
    loss_function = nn.CrossEntropyLoss()
    GET_PERPLEXITY = False
    PREDICT_WORD = False
    context = "he is a"
    GENERATE = False
    number_words = 300
    temp = 1
    ##########################################################################

    # # get test data
    corpus = Corpus(DATA_FILE, CUDA)
    vocab_size = len(corpus.dictionary)
    print("V", vocab_size)
    test_data = batchify(corpus.test, BSZ, CUDA)

    if GET_PERPLEXITY or args.perplexity:
        print("Perplexity")
        print(math.exp(evaluate(model, corpus, loss_function, test_data, CUDA, BSZ, BPTT)))
    elif PREDICT_WORD:
        print("Context:")
        print(context)
        print("Next word:")
        print(predict_word(corpus, model, context, CUDA))
    elif GENERATE:
        generate(corpus, CUDA, temp, number_words, 1, 1)
    else:
        print("Set either GET_PERPLEXITY, PREDICT_WORD or GENERATE to true")

    # test_sentence1 = "an earthquake struck northern california killing more than N people"
    # test_sentence2 = "he sits down at the piano and plays"
    # test_sentence3 = "conservative party fails to secure a majority resulting in a hung parliament"
    # print(test_sentence1)
    # word_dependencies(model, corpus, test_sentence1.split(),
    #                   "sum", args.name+"f1.svg", CUDA, True)
    # print(test_sentence2)
    # word_dependencies(model, corpus, test_sentence2.split(),
    #                   "sum", args.name+"f2.svg", CUDA, True)
    # print(test_sentence3)
    # word_dependencies(model, corpus, test_sentence3.split(),
    #                   "sum", args.name+"f3.svg", CUDA, True)

    max_lengths_med, med_lengths_med, maxarc_lengths, medianarc_lengths = analysisArcLength(args.datafile, args.name,
                                                                                            model, corpus, args.norm, CUDA, 1000000)

    filebase = args.norm+'_'+args.datafile[-5:-1]+'_'+args.name
    save_obj(max_lengths_med, 'max_l_med_'+filebase)
    save_obj(med_lengths_med, 'med_l_med_'+filebase)
    save_obj(maxarc_lengths, 'all_max_'+filebase)
    save_obj(medianarc_lengths, 'all_med_'+filebase)
