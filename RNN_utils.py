import numpy as np


def window_subsample(features, labels, window_length=5):

    num_data = features.shape[0]
    num_features = features.shape[1]
    num_windows = num_data - window_length + 1

    data = []
    labels = labels[(window_length - 1):]

    for i in range(num_windows):
        sample = features[i:i+window_length]
        data.append(sample)

    data = np.asarray(data)

    return data, labels
