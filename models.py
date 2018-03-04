import math
import tensorflow as tf
from utils import conv_size

from config import slice_size, channels
import numpy as np


class FcStack:
    def __init__(self, input_size, coding_size, activation=tf.nn.elu, depth=1):
        self.input_size = input_size
        self.slice_size = input_size
        self.coding_size = coding_size
        self.w_list = [
            tf.Variable(tf.random_normal(
                [input_size, input_size]
            ) * np.sqrt(2.0 / (input_size + input_size)))
            for _ in range(depth - 1)
        ]
        self.w_list.append(
            tf.Variable(tf.random_normal(
                [input_size, coding_size]
            ) * np.sqrt(2.0 / (input_size + coding_size)))
        )

        self.b_list = [
            tf.Variable(tf.zeros(input_size))
            for _ in range(depth - 1)
        ]
        self.b_list.append(
            tf.Variable(tf.zeros(coding_size))
        )

        self.activation_list = [
            activation for _ in range(depth - 1)
        ]

        self.activation_list.append(None)

    @property
    def n_features(self):
        return 613

    def preprocess(self, x):
        assert (channels == 1)
        x = tf.reshape(x, (-1, slice_size))
        rfft = tf.spectral.rfft(x)
        re = tf.real(rfft)
        im = tf.imag(rfft)
        return tf.square(re) + tf.square(im)

    def prepare(self, x):
        return tf.reshape(x, (-1, self.n_features))

    def predict(self, prepared_inputs):
        codings = prepared_inputs
        for i in range(len(self.w_list)):
            w = self.w_list[i]
            b = self.b_list[i]
            activation = self.activation_list[i]
            codings = tf.matmul(codings, w) + b
            mean, var = tf.nn.moments(codings, axes=[0, ])
            codings = tf.nn.batch_normalization(codings, mean, var, None,
                                                None, 1e-3)
            if activation is not None:
                codings = activation(codings)
        return codings

    def cost(self, prepared_inputs, y):
        logits = self.predict(prepared_inputs)
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=y)
        )

    def training_feeds(self):
        return {}

    def testing_feeds(self):
        return {}
