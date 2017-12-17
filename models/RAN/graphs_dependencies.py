from predict_word import load_obj
import statistics
import numpy as np
import matplotlib.pyplot as plt


def medians_per_sentence_length(data, groups, normalized=False):

    medians = []
    for key in groups:
        if str(key) in list(data.keys()):
            divide_by = 1
            if normalized:
                divide_by = key
            medians.append(statistics.median(data[str(key)]) / divide_by)
        else:
            medians.append(0)

    return medians


def grouped_barchart(data1, data2, data3):

    # Setting the positions and width for the bars
    pos = [i for i in range(2, 55, 5)]
    width = 1

    medians_data1 = medians_per_sentence_length(data1, pos)
    medians_data2 = medians_per_sentence_length(data2, pos)
    medians_data3 = medians_per_sentence_length(data3, pos)

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos,
            # using df['pre_score'] data,
            medians_data1,
            # of width
            width,
            # with alpha 0.5
            alpha=1,
            # with color
            # with label the first value in first_name
            label='1-layer S-LSTM')

    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos],
            # using df['mid_score'] data,
            medians_data2,
            # of width
            width,
            # with alpha 0.5
            alpha=1,
            # with color
            # with label the second value in first_name
            label='First Layer of 2-layer S-LSTM')

    # Create a bar with post_score data,
    # in position pos + some width buffer,
    plt.bar([p + width * 2 for p in pos],
            # using df['post_score'] data,
            medians_data3,
            # of width
            width,
            # with alpha 0.5
            alpha=1,
            # with color
            # with label the third value in first_name
            label='Second Layer of 2-layer S-LSTM')

    # Set the y axis label
    ax.set_ylabel('Dependency Length')
    ax.set_xlabel('Sentence Length')

    # Set the chart's title
    ax.set_title(
        'Median of Average Dependency Length per Sentence Length\nOne Layer Simplified LSTM - Forget Gates')

    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels([str(i) for i in pos])

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 4)

    # Adding the legend and showing the plot
    plt.legend()
    ax.yaxis.grid()
    plt.show()


def barchartLayers(data1, data2, data3):

    # Setting the positions and width for the bars
    pos = [i for i in range(2, 55, 5)]
    width = 1

    medians_data1 = medians_per_sentence_length(data1, pos)
    medians_data2 = medians_per_sentence_length(data2, pos)
    medians_data3 = medians_per_sentence_length(data3, pos)

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos,
            # using df['pre_score'] data,
            medians_data1,
            # of width
            width,
            # with alpha 0.5
            alpha=1,
            # with color
            # with label the first value in first_name
            label='1-layer S-LSTM')

    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos],
            # using df['mid_score'] data,
            medians_data2,
            # of width
            width,
            # with alpha 0.5
            alpha=1,
            # with color
            # with label the second value in first_name
            label='First Layer of 2-layer S-LSTM')

    # Create a bar with post_score data,
    # in position pos + some width buffer,
    plt.bar([p + width * 2 for p in pos],
            # using df['post_score'] data,
            medians_data3,
            # of width
            width,
            # with alpha 0.5
            alpha=1,
            # with color
            # with label the third value in first_name
            label='Second Layer of 2-layer S-LSTM')

    # Set the y axis label
    ax.set_ylabel('Dependency Lengths')
    ax.set_xlabel('Sentence Lengths')

    # Set the chart's title
    ax.set_title(
        'Median Dependency Length per Different Models of S-LSTM')

    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels([str(i) for i in pos])

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 4)

    # Adding the legend and showing the plot
    plt.legend()
    ax.yaxis.grid()
    plt.show()

pickle_location_max = 'train_max/'
pickle_location_sum = 'train_sum/'
pickle_location_l2 = 'train_l2/'

penn_loc = 'train_sum/'
zech_loc = 'czech_dep/'

files_penn_l2 = [
    'all_max_l2_penn_one_layer',
    'all_med_l2_penn_one_layer',
    'max_l_med_l2_penn_one_layer',
    'med_l_med_l2_penn_one_layer'
]
files_penn_sum = [
    'all_max_sum_penn_one_layer',
    'all_med_sum_penn_one_layer',
    'max_l_med_sum_penn_one_layer',
    'med_l_med_sum_penn_one_layer'
]
files_penn_max = [
    'all_max_max_penn_one_layer',
    'all_med_max_penn_one_layer',
    'max_l_med_max_penn_one_layer',
    'med_l_med_max_penn_one_layer'
]

pickle_location_2_all = 'dependencies_penn2_all/'
pickle_location_1_all = 'dependencies_penn1_all/'


files_penn_2_all = [
    'all_max_sum_penn_two_layer',
    'all_med_max_penn_two_layer'
]

files_penn_1_all = [
    'all_max_sum_penn_one_layer',
    'all_med_max_penn_one_layer'
]

lvl2_data = load_obj(pickle_location_2_all + files_penn_2_all[0])
lvl1_data = load_obj(pickle_location_1_all + files_penn_1_all[0])

barchartLayers(lvl1_data[0], lvl2_data[0], lvl2_data[1])

files_zech_sum = [
    'all_max_sum_zech_one_layer',
    'all_med_sum_zech_one_layer',
    'max_l_med_sum_zech_one_layer',
    'med_l_med_sum_zech_one_layer'
]

max_data = load_obj(pickle_location_max + files_penn_max[1])
sum_data = load_obj(pickle_location_sum + files_penn_sum[1])
l2_data = load_obj(pickle_location_l2 + files_penn_l2[1])
for key, value in max_data.items():
    print(key)
    print(sorted(value))
    print()
grouped_barchart(max_data[0], sum_data[0], l2_data[0])

penn_data = load_obj(penn_loc + files_penn_sum[1])
czech_data = load_obj(zech_loc + files_zech_sum[1])
