from scipy import signal
import numpy as np


def sampling(data, start, step):
    """
    Choose sample from data.
    """
    length = data.size()[2]
    sample = set(range(start, length, step))
    rest = set(range(length))
    data_ind = list(rest.difference(sample))
    return data_ind


def interpolation(data, data_len, start):
    """
    Interpolate the sampled data.
    """
    x, y, z = data.size()
    data_new = np.zeros((x, y, data_len))
    data_ind = sampling(data, start, 8)
    data = data[:, :, data_ind]
    for i in range(x):
        for j in range(y):
            data_new[i, j] = signal.resample(data[i, j], data_len)
    return data_new
