import math


def conv_size(input_width, filter_width, stride, padding):
    if padding == 'SAME':
        return math.ceil(float(input_width) / float(stride))
    if padding == 'VALID':
        return math.ceil(float(input_width - filter_width) / float(
            stride)) + 1
