import math
import numpy as np


def conv_size(input_width, filter_width, stride, padding):
    if padding == 'SAME':
        return math.ceil(float(input_width) / float(stride))
    if padding == 'VALID':
        return math.ceil(float(input_width - filter_width) / float(
            stride)) + 1


def shuffle(*arrays):
    assert (all(x.shape[0] == arrays[0].shape[0] for x in arrays))
    indices = np.arange(0, arrays[0].shape[0])
    np.random.shuffle(indices)
    shuffled = tuple(
        array[indices] for array in arrays
    )
    if len(shuffled) == 1:
        return shuffled[0]
    else:
        return shuffled


def whole_multiple(x, y):
    return x - x % y
