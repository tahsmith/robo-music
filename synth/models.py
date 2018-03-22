import tensorflow as tf

from .config import slice_size, channels
import numpy as np


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

        self.encoding_activation_list.append(None)
        self.decoding_activation_list = \
            list(reversed(self.encoding_activation_list))

    def fc_layer(self, input_size, coding_size):
        return tf.Variable(tf.random_normal(
            [input_size, coding_size]
        ) * np.sqrt(2.0 / (input_size + coding_size)))

    @property
    def n_features(self):
        return 1226

    def preprocess(self, x):
        assert (channels == 1)
        x = tf.reshape(x, (-1, slice_size))
        rfft = tf.spectral.rfft(x)
        mod = tf.abs(rfft)
        arg = tf.angle(rfft)
        return tf.concat((mod, arg), axis=1)

    def prepare(self, x):
        return tf.reshape(x, (-1, self.n_features))

    def encoder(self, prepared_inputs):
        codings = prepared_inputs
        for i in range(len(self.w_encode_list)):
            w = self.w_encode_list[i]
            b = self.b_encode_list[i]
            activation = self.encoding_activation_list[i]
            codings = tf.matmul(codings, w) + b
            # mean, var = tf.nn.moments(codings, axes=[0,])
            # codings = tf.nn.batch_normalization(codings, mean, var, None,
            #                                     None, 1e-3)
            if activation is not None:
                codings = activation(codings)
        return codings

    def decoder(self, codings):
        decoded = codings
        for i in reversed(range(len(self.w_encode_list))):
            w = self.w_decode_list[i]
            b = self.b_decode_list[i]
            activation = self.decoding_activation_list[i]
            decoded = tf.matmul(decoded, tf.transpose(w)) + b
            # mean, var = tf.nn.moments(decoded, axes=[0,])
            # decoded = tf.nn.batch_normalization(decoded, mean, var, None,
            #                                     None, 1e-3)
            if activation is not None:
                decoded = activation(decoded)
        return decoded

    def cost(self, prepared_inputs, _):
        reconstructed = self.decoder(self.encoder(prepared_inputs))
        return tf.reduce_mean(tf.square(prepared_inputs - reconstructed))

    def predict(self, batches):
        return self.decoder(self.encoder(batches))

    def training_feeds(self):
        return {}

    def testing_feeds(self):
        return {}
