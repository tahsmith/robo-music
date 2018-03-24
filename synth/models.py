import tensorflow as tf

from .config import slice_size, channels, feature_size
import numpy as np
from utils import conv_size


class FcStack:
    def __init__(self,
                 input_size,
                 coding_size,
                 activation=tf.nn.elu,
                 depth=1,
                 reuse=True
                 ):
        self.input_size = input_size
        self.slice_size = input_size
        self.coding_size = coding_size
        self.w_encode_list = [
            self.fc_layer(input_size, input_size)
            for _ in range(depth - 1)
        ]
        self.w_encode_list.append(
            self.fc_layer(input_size, coding_size)
        )

        self.b_encode_list = [
            tf.Variable(tf.zeros(input_size))
            for _ in range(depth - 1)
        ]
        self.b_encode_list.append(
            tf.Variable(tf.zeros(coding_size))
        )
        self.b_decode_list = [
            tf.Variable(tf.zeros(input_size))
            for _ in range(depth)
        ]
        if reuse:
            self.w_decode_list = self.w_encode_list
        else:
            self.w_decode_list = [
                self.fc_layer(input_size, input_size)
                for _ in range(depth - 1)
            ]
            self.w_decode_list.append(
                self.fc_layer(input_size, coding_size)
            )

        self.encoding_activation_list = [
            activation for _ in range(depth - 1)
        ]
        self.encoding_activation_list.append(tf.nn.sigmoid)

        self.decoding_activation_list = [None] + [
            activation for _ in range(depth - 1)
        ]

        self.noise = tf.placeholder(dtype=tf.float32)
        self.dropout_rate = tf.placeholder(dtype=tf.float32)

    def fc_layer(self, input_size, coding_size):
        return tf.Variable(tf.random_normal(
            [input_size, coding_size]
        ) * np.sqrt(2.0 / (input_size + coding_size)))

    @property
    def n_features(self):
        return feature_size

    def generate_features(self, x):
        assert (channels == 1)
        x = tf.reshape(x, (-1, slice_size))
        rfft = tf.spectral.rfft(x)
        arg = tf.angle(rfft)
        mag = tf.abs(rfft)
        return tf.concat((arg, mag), axis=1)

    def data_from_features(self, x):
        arg = x[:, :feature_size // 2]
        mag = x[:, feature_size // 2:]

        re = mag * tf.cos(arg)
        im = mag * tf.sin(arg)
        return tf.spectral.irfft(1j * tf.cast(im, tf.complex64) + tf.cast(re,
                                                                          tf.complex64))

    def prepare(self, x):
        return tf.reshape(x, (-1, self.n_features))

    def encoder(self, prepared_inputs):
        codings = prepared_inputs + tf.random_normal(tf.shape(
            prepared_inputs), self.noise)
        for i in range(len(self.w_encode_list)):
            w = self.w_encode_list[i]
            b = self.b_encode_list[i]
            activation = self.encoding_activation_list[i]
            codings = tf.matmul(codings, w) + b
            if activation is not None:
                mean, var = tf.nn.moments(codings, axes=[0])
                codings = tf.nn.batch_normalization(codings, mean, var, None,
                                                    None, 1e-3)
                codings = activation(codings)

        return tf.layers.dropout(codings, self.dropout_rate)

    def decoder(self, codings):
        decoded = codings
        for i in reversed(range(len(self.w_encode_list))):
            w = self.w_decode_list[i]
            b = self.b_decode_list[i]
            activation = self.decoding_activation_list[i]
            decoded = tf.matmul(decoded, tf.transpose(w)) + b
            if activation is not None:
                mean, var = tf.nn.moments(decoded, axes=[0])
                decoded = tf.nn.batch_normalization(decoded, mean, var, None,
                                                    None, 1e-3)
                decoded = activation(decoded)
        return decoded

    def cost(self, prepared_inputs, _):
        reconstructed = self.decoder(self.encoder(prepared_inputs))
        return tf.reduce_mean(tf.square(prepared_inputs - reconstructed))

    def predict(self, batches):
        return self.decoder(self.encoder(batches))

    def training_feeds(self):
        return {
            self.noise: 0.0,
            self.dropout_rate: 0.0
        }

    def testing_feeds(self):
        return {
            self.noise: 0.0,
            self.dropout_rate: 0.0
        }


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
            **self.fc_stack.training_feeds()
        }

    def testing_feeds(self):
        return {
            self.keep_prob: 1.0,
            **self.fc_stack.testing_feeds()
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

        output = tf.reshape(output, (-1, feature_size))
        return output

    def cost(self, batches, _):
        reconstructed = self.decoder(self.encoder(batches))
        return tf.reduce_mean(tf.square(batches - reconstructed))

    def predict(self, batches):
        return self.decoder(self.encoder(batches))

    @property
    def n_features(self):
        return feature_size
