from predict_word import load_obj
import statistics
import numpy as np
import matplotlib.pyplot as plt


def medians_per_sentence_length(data, groups, normalized=True):

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

    groups = sorted([int(d) for d in list(data1.keys())])
    groups = np.array(groups)
    medians_data1 = medians_per_sentence_length(data1, groups)
    medians_data2 = medians_per_sentence_length(data2, groups)
    medians_data3 = medians_per_sentence_length(data3, groups)

    fig, ax = plt.subplots()
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(groups - bar_width, medians_data1, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='max')

    rects2 = plt.bar(groups, medians_data2, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='sum')

    rects2 = plt.bar(groups + bar_width, medians_data3, bar_width,
                     alpha=opacity,
                     color='g',
                     error_kw=error_config,
                     label='l2')

    plt.xlabel('Sentence Length')
    plt.ylabel('Normalized Dependency Distance')
    plt.title('Word Dependencies Per Sentence Length by 1-layer RAN\nNormalized By Sentence Length')
    plt.xticks(groups + bar_width / 2, tuple([str(i) for i in groups]))
    plt.legend()

    plt.tight_layout()
    plt.show()


pickle_location_max = 'dependencies_penn1_max/'
pickle_location_sum = 'dependencies_penn1_sum/'
pickle_location_l2 = 'dependencies_penn1_l2/'
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

max_data = load_obj(pickle_location_max + files_penn_max[0])
sum_data = load_obj(pickle_location_sum + files_penn_sum[0])
l2_data = load_obj(pickle_location_l2 + files_penn_l2[0])
print(max_data)
grouped_barchart(max_data, sum_data, l2_data)