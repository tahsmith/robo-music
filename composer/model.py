import tensorflow as tf
from tensorflow.contrib import rnn
import synth.config
from . import config

class Model:
    def __init__(self, desynth, coding_size, neuron_count):
        cell = rnn.OutputProjectionWrapper(
            rnn.DropoutWrapper(
                rnn.LSTMCell(num_units=neuron_count,
                             initializer=tf.variance_scaling_initializer(),
                             activation=tf.nn.elu,
                             ),
                input_keep_prob=0.7
            ),
            output_size=coding_size,
        )

        self.outputs, self.states = tf.nn.dynamic_rnn(cell, desynth,
                                                 dtype=tf.float32)


codings = tf.placeholder(tf.float32,
                   [None, config.n_steps, config.n_inputs],
                   name='codings')

model = Model(codings, synth.config.coding_size, config.n_neurons)
outputs = model.outputs