import tensorflow as tf

from .config import slice_size, channels
import numpy as np
from utils import conv_size


def conv1d(input, filters, kernel_size, stride):
    return tf.layers.conv2d(
        input,
        filters,
        kernel_size=kernel_size,
        strides=(1, stride),
        padding="same",
    )


def conv1d_transpose(input, filters, kernel_size, stride):
    return tf.layers.conv2d_transpose(
        input,
        filters,
        kernel_size=kernel_size,
        strides=(1, stride),
        padding="same"
    )


def layer(input_, filters, kernel_size, stride):
    filter_ = conv1d(input_, filters, kernel_size, stride)
    gate = conv1d(input_, filters, kernel_size, stride)

    output = tf.tanh(filter_) * tf.sigmoid(gate)

    mean, var = tf.nn.moments(output, axes=[0])
    output = tf.nn.batch_normalization(output, mean, var, None,
                                       None, 1e-3)

    return output


def layer_transpose(input_, filters, kernel_size, stride):
    filter_ = conv1d_transpose(input_, filters, kernel_size, stride)
    gate = conv1d_transpose(input_, filters, kernel_size, stride)

    output = tf.tanh(filter_) * tf.sigmoid(gate)

    return output


class DeepConvModel:
    def __init__(self, slice_size, widths, strides, channels, activation,
                 fc_stack):
        self.n_layers = len(widths)
        self.slice_size = slice_size
        self.widths = [slice_size] + widths
        self.strides = [None] + strides
        self.channels = [1] + channels
        self.activations = [None] + [activation] * self.n_layers

        n_layers = len(widths)
        self.output_widths = [slice_size]
        self.output_depths = [1]
        for i in range(1, n_layers + 1):
            self.output_widths.append(
                conv_size(self.output_widths[i - 1], self.widths[i],
                          self.strides[i], 'SAME')
            )
            self.output_depths.append(self.channels[i])

        self.fc_stack = fc_stack

        self.keep_prob = tf.placeholder(tf.float32)

    def training_feeds(self):
        return {
            self.keep_prob: 0.6,
        }

    def testing_feeds(self):
        return {
            self.keep_prob: 1.0,
        }

    def prepare(self, batch):
        return batch

    def encoder(self, prepared_inputs):
        prepared_inputs = tf.reshape(prepared_inputs, (-1, 1, self.n_features,
                                                       1))
        output = prepared_inputs

        for i in range(1, self.n_layers + 1):
            output = layer(
                output,
                self.channels[i],
                self.widths[i],
                self.strides[i]
            )

        output = tf.reshape(
            output,
            [-1, self.output_widths[-1] *
             self.output_depths[-1]]
        )

        output = tf.nn.dropout(output, self.keep_prob)

        if self.fc_stack is not None:
            output = self.fc_stack.encoder(output)

        return output

    def decoder(self, codings):
        output = codings
        if self.fc_stack is not None:
            output = self.fc_stack.decoder(output)

        output = tf.reshape(
            output,
            [-1, 1, self.output_widths[-1], self.output_depths[-1]]
        )

        n_batches = tf.shape(codings)[0]
        for i in reversed(range(1, self.n_layers + 1)):
            output = layer_transpose(
                output,
                self.channels[i - 1],
                self.widths[i],
                self.strides[i]
            )

        output = tf.reshape(output, (-1, slice_size, channels))
        return output

    def cost(self, batches, _):
        reconstructed = self.decoder(self.encoder(batches))
        return tf.reduce_mean(tf.square(batches - reconstructed))

    def predict(self, batches):
        return self.decoder(self.encoder(batches))

    @property
    def n_features(self):
        return slice_size
