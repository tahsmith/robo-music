import math
import tensorflow as tf
from utils import conv_size
timeslice_size = 306
samples_per_second = 11025
import numpy as np


class Model:
    def reconstructed(self, batches):
        return self.decoder(self.encoder(batches))

    def training_feeds(self):
        return {}

    def testing_feeds(self):
        return {}


class LinearModel(Model):
    def __init__(self, slice_size, coding_size):
        self.slice_size = slice_size
        self.coding_size = coding_size
        self.w = tf.Variable(tf.random_normal(
            [slice_size, self.coding_size]
        ) * np.sqrt(2.0 / (coding_size + slice_size)))
        self.b = tf.Variable(tf.zeros(self.coding_size))
        self.keep_prob = tf.placeholder(tf.float32)

    def training_feeds(self):
        return {
            self.keep_prob: 0.8
        }

    def testing_feeds(self):
        return {
            self.keep_prob: 1.0
        }

    @property
    def variables(self):
        return (
            self.w,
            self.b
        )

    def prepare(self, x):
        return tf.reshape(x, (-1, self.slice_size))

    def encoder(self, prepared_inputs):
        return tf.matmul(prepared_inputs, self.w) + self.b

    def decoder(self, codings):
        return tf.matmul(codings - self.b, tf.transpose(self.w))

    def cost(self, prepared_inputs):
        reconstructed = self.decoder(self.encoder(prepared_inputs))
        return tf.reduce_mean(tf.square(prepared_inputs - reconstructed))


class ConvModel(Model):
    def __init__(self, input_width, filter_width, stride,
                 filter_out_channels,
                 padding):
        self.padding = padding
        self.out_channels = filter_out_channels
        self.stride = stride
        self.slice_size = input_width
        self.width = filter_width
        self.w = tf.Variable(
            tf.random_normal((1, filter_width, 1, filter_out_channels)) *
            np.sqrt(2.0 / (input_width + filter_width)),
            name='w'
        )
        self.b = tf.Variable(
            tf.zeros((filter_out_channels,)),
            name='b'
        )

    @property
    def variables(self):
        return (
            self.w,
            self.b
        )

    @property
    def coding_size(self):
        return conv_size(self.slice_size, self.width, self.stride,
                         self.padding) * self.out_channels

    def prepare(self, batch):
        return tf.reshape(batch, (-1, 1, self.slice_size, 1))

    def encoder(self, prepared_inputs):
        return tf.nn.conv2d(
            prepared_inputs,
            self.w,
            strides=[1, 1, self.stride, 1],
            padding=self.padding
        ) + self.b

    def decoder(self, codings):
        n_batches = tf.shape(codings)[0]
        return tf.nn.conv2d_transpose(
            codings - self.b,
            self.w,
            [n_batches, 1, self.slice_size, 1],
            strides=[1, 1, self.stride, 1],
            padding='SAME'
        )

    def cost(self, batches):
        reconstructed = self.reconstructed(batches)
        return tf.reduce_mean(tf.square(batches - reconstructed))


def model(inputs, width, depth, batches):
    conv_layer_config = [
        # (1, 1, 2, tf.nn.relu),
        (5, 1, 1, None),
    ]

    reshaped_inputs = tf.reshape(
        inputs,
        (batches, 1, width, depth)
    )
    conv_kernels = []
    conv_biases = []
    input_shapes = [tf.shape(reshaped_inputs)[2]]
    input_depths = [depth]
    output_depths = []
    strides = []
    widths = []
    activations = []
    encode_ops = [reshaped_inputs]
    for config in conv_layer_config:
        width, stride, output_depth, activation = config
        output_depths.append(output_depth)
        widths.append(width)
        strides.append(stride)
        activations.append(activation)
        conv_kernels.append(
            tf.Variable(tf.random_normal((1, width, input_depths[-1],
                                          output_depth))))
        conv_biases.append(
            tf.Variable(tf.zeros((output_depth,)))
        )
        input_depths.append(output_depth)
        op = tf.nn.conv2d(
            encode_ops[-1],
            conv_kernels[-1],
            strides=[1, 1, stride, 1],
            padding='SAME'
        ) + conv_biases[-1]
        if activation:
            op = activation(op)
        encode_ops.append(op)
        input_shapes.append(tf.shape(encode_ops[-1])[2])

    input_shapes = input_shapes[:-1]
    input_depths = input_depths[:-1]

    flatten = tf.reshape(encode_ops[-1], (batches, -1))
    # Add fully connected layer

    fc_layer_1_w = tf.Variable(tf.random_normal([int(flatten.shape[1]),
                                                 timeslice_size // 4]))
    fc_layer_1_b = tf.Variable(tf.zeros(timeslice_size // 4))
    fc_layer_1 = tf.matmul(flatten, fc_layer_1_w) + fc_layer_1_b
    encoded = fc_layer_1

    fc_layer_1_rev = tf.matmul(encoded - fc_layer_1_b,
                               tf.transpose(fc_layer_1_w))

    encoded_reshaped = tf.reshape(fc_layer_1_rev, (batches, 1, -1,
                                                   output_depths[-1]))
    output = encoded_reshaped
    for i in reversed(range(len(conv_kernels))):
        if activations[i]:
            output = activations[i](output)
        output = tf.nn.conv2d_transpose(
            output - conv_biases[i],
            conv_kernels[i],
            [batches, 1, input_shapes[i], input_depths[i]],
            strides=[1, 1, strides[i], 1],
            padding='SAME'
        )
    decoded = tf.reshape(output, (batches, -1, depth))

    return encoded, decoded