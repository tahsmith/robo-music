import tensorflow as tf

from .config import slice_size, channels
import numpy as np
from utils import conv_size


class DeepConvModel:
    def __init__(self, slice_size, widths, strides, channels,
                 paddings, activations, fc_stack):
        self.n_layers = len(widths)
        self.slice_size = slice_size
        self.widths = [slice_size] + widths
        self.strides = [None] + strides
        self.channels = [1] + channels
        self.paddings = [None] + paddings
        self.activations = [None] + activations

        n_layers = len(widths)
        self.output_widths = [slice_size]
        self.output_depths = [1]
        for i in range(1, n_layers + 1):
            self.output_widths.append(
                conv_size(self.output_widths[i - 1], self.widths[i],
                          self.strides[i], self.paddings[i])
            )
            self.output_depths.append(self.channels[i])
        self.w_list = [
            tf.Variable(
                tf.random_normal((1, self.widths[i], self.output_depths[i - 1],
                                  self.channels[i])) *
                np.sqrt(2.0 / (self.output_widths[i - 1] + self.widths[i])),
                name=f'w{i}'
            ) for i in range(1, n_layers + 1)
        ]
        self.b_list = [
            tf.Variable(
                tf.zeros((self.channels[i],)),
                name=f'b{i}'
            ) for i in range(1, n_layers + 1)
        ]

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

    @property
    def variables(self):
        for i in range(len(self.w_list)):
            yield self.w_list[i]
            yield self.b_list[i]

    @property
    def coding_size(self):
        return self.output_widths[-1]

    def prepare(self, batch):
        return batch

    def encoder(self, prepared_inputs):
        prepared_inputs = tf.reshape(prepared_inputs, (-1, 1, self.n_features,
                                                      1))
        output = prepared_inputs
        for i in range(1, self.n_layers + 1):
            output = tf.nn.conv2d(
                output,
                self.w_list[i - 1],
                strides=[1, 1, self.strides[i], 1],
                padding=self.paddings[i]
            ) + self.b_list[i - 1]

            mean, var = tf.nn.moments(output, axes=[0])
            output = tf.nn.batch_normalization(output, mean, var, None,
                                                None, 1e-3)

            if self.activations[i] is not None:
                output = self.activations[i](output)
                output = tf.nn.dropout(output, self.keep_prob)

        output = tf.reshape(
            output,
            [-1, self.output_widths[-1] *
             self.output_depths[-1]]
        )

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
            output = tf.nn.conv2d_transpose(
                output - self.b_list[i - 1],
                self.w_list[i - 1],
                [n_batches, 1, self.output_widths[i - 1],
                 self.output_depths[i - 1]],
                strides=[1, 1, self.strides[i], 1],
                padding='SAME'
            )
            if self.activations[i] is not None:
                output = self.activations[i](output)

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
