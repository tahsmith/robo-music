import math

import numpy as np


def conv_size(input_width, filter_width, stride, padding):
    if padding == 'SAME':
        return math.ceil(float(input_width) / float(stride))
    if padding == 'VALID':
        return math.ceil(float(input_width - filter_width) / float(
            stride)) + 1

    raise ValueError


def atrous_conv_size(input_width, filter_width, dilation, padding):
    if padding == "SAME":
        return input_width
    if padding == "VALID":
        return input_width - 2 * (filter_width - 1)

    raise ValueError


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


def normalise_to_int_range(x, dtype):
    int_info = np.iinfo(dtype)
    max_int = int_info.max
    min_int = int_info.min
    max_val = np.max(x)
    min_val = np.min(x)

    x = (x - min_val) / (max_val - min_val)
    x = x * (max_int - min_int) + min_int
    x = np.rint(x).astype(dtype)
    return x


def dilate_zero_order_hold(x, padding):
    if len(x.shape) != 2:
        raise NotImplementedError()

    return np.tile(x, (1, padding)).reshape((x.shape[0] * padding, x.shape[1]))
