import pandas as pd
import numpy as np


def window_subsample(features, labels, window_length=5):

    num_data = features.shape[0]
    num_features = features.shape[1]
    num_windows = num_data - window_length - 4

    data = []
    future = []
    labels = labels[window_length:len(labels)-5]

    np_features = features.values

    for i in range(num_windows - 1):
        sample = np_features[i:i+window_length]
        data.append(sample)

    for i in range(num_windows - 1):
        curr_data = features.iloc[i + window_length, np.r_[2:7, 31:36, 37:42, 43:48]].values
        # pd_1 = features.iloc[i + window_length+1, np.r_[3:7, 32:36, 38:42, 44:48]].values
        # pd_2 = features.iloc[i + window_length+2, np.r_[4:7, 33:36, 39:42, 45:48]].values
        # pd_3 = features.iloc[i + window_length+3, np.r_[5:7, 34:36, 40:42, 46:48]].values
        # pd_4 = features.iloc[i + window_length+4, np.r_[6:7, 35:36, 41:42, 47:48]].values

        # future_row = np.concatenate((curr_data, pd_1, pd_2, pd_3, pd_4))
        future.append(curr_data)

    data = np.asarray(data)
    future = np.asarray(future)

    return data, labels, future
