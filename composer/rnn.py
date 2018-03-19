from sys import argv

import math
from tensorflow.contrib import rnn

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from contexttimer import Timer

from synth import prepare

slice_size = 612
fft_size = slice_size // 2 + 1
steps_seconds = 2.0
n_steps = math.ceil(steps_seconds * prepare.samples_per_second / slice_size)
n_inputs = 2 * fft_size
n_neurons = 20
n_outputs = 2 * fft_size

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='X')
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs], name='y')

cell = rnn.OutputProjectionWrapper(
    rnn.DropoutWrapper(
        rnn.LSTMCell(num_units=n_neurons,
                     initializer=tf.variance_scaling_initializer(),
                     activation=tf.nn.elu,
                     ),
        input_keep_prob=0.7
    ),
    output_size=n_outputs,
)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 2000
n_iterations = 150
batch_size = 1000

t_min, t_max = 0, 30
resolution = 1 / prepare.samples_per_second


def normalise(x):
    mean = np.mean(x)
    std = np.std(x)
    print(mean, std)
    x = (x - mean) / (std + 1e-3)
    clipped_size = x.shape[0] - x.shape[0] % slice_size
    x = x[0:clipped_size]
    return x


def waveform_to_features(x):
    x = x.reshape((-1, slice_size))
    x = np.fft.rfft(x, axis=1)
    re = np.real(x)
    im = np.imag(x)
    x = np.concatenate((re, im), axis=1)
    return x


def features_to_waveform(x):
    re = x[:, :fft_size]
    im = x[:, fft_size:]
    x = re + 1j * im
    slices = np.fft.irfft(x, axis=1)
    waveform = slices.reshape((-1,))
    return waveform


def next_batch(X_all, batch_size, n_steps):
    start_indices = np.random.randint(0, X_all.shape[0] - n_steps,
                                      size=(batch_size, 1))
    indices = start_indices + np.arange(0, n_steps + 1)
    return (
        X_all[indices[:, :-1], :].reshape(batch_size, n_steps, n_inputs),
        X_all[indices[:, 1:], :].reshape(batch_size, n_steps, n_inputs)
    )


def generate(X_all):
    batch = np.random.randn(1, n_steps, n_inputs)
    wave_form = np.zeros((0, n_inputs))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './sess')
        for i in range(10 * prepare.samples_per_second // slice_size):
            new_point = sess.run(outputs,
                                 feed_dict={X: batch})
            batch = np.concatenate((batch[:, 1:, :], new_point[:, -1:, :]),
                                   axis=1)
            new_point = new_point[0, -1, :].reshape((1, n_inputs))
            wave_form = np.concatenate((wave_form, new_point))

            print(f'{i * slice_size / prepare.samples_per_second}s')

    wave_form = features_to_waveform(wave_form)
    print(wave_form.shape)
    return wave_form


class EarlyTermination(Exception):
    pass


def main():
    T_all = np.arange(0, t_max, resolution)
    theta = 220 * T_all + 110 * np.cos(T_all * 2 * np.pi) / (2 * np.pi)
    X_all = np.sin(theta * 2 * np.pi)
    # X_all = np.random.randn(prepare.samples_per_second * 10)
    # X_all = np.fromfile(
    #     '/home/tahsmith/src/audio-synth/data/bensound-goinghigher.raw',
    #     dtype='<i2'
    # )
    X_all = normalise(X_all)
    X_all = waveform_to_features(X_all)
    print(X_all.dtype, X_all.shape)

    saver = tf.train.Saver()
    if len(argv) > 1:
        with Timer() as t:
            with tf.Session() as sess:
                init.run()
                saver.restore(sess, './sess')
                try:
                    for epoc in range(n_epochs):
                        for iteration in range(n_iterations):
                            X_batch, y_batch = next_batch(X_all, batch_size,
                                                          n_steps)
                            sess.run(training_op,
                                     feed_dict={X: X_batch, y: y_batch})
                            if iteration % 10 == 0:
                                loss_eval = loss.eval(
                                    feed_dict={X: X_batch, y: y_batch})
                                print(iteration, "t = ", t.elapsed, "Loss: ",
                                      loss_eval)
                                saver.save(sess, './sess')
                                if loss_eval < 10:
                                    raise EarlyTermination
                                if t.elapsed > 2500:
                                    raise EarlyTermination
                except EarlyTermination:
                    pass
    else:
        waveform = generate(X_all)
        (waveform * 2 ** 14).astype('int16').tofile('generated.raw',
                                                    format='<i2')


if __name__ == '__main__':
    main()
