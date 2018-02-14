from glob import glob
from random import shuffle

import tensorflow as tf
import numpy as np
from tensorflow.contrib import ffmpeg

from model import LinearModel


def optimiser(model, batches):
    batches = model.prepare(batches)
    optimiser = tf.train.AdamOptimizer()
    cost = model.cost(batches)
    codings = model.encoder(batches)
    reconstructed = model.decoder(codings)
    abs_error = tf.reduce_mean(tf.abs(batches - reconstructed))
    avg = tf.reduce_mean(tf.abs(batches + reconstructed)) / 2
    relative_error = abs_error / avg
    op = optimiser.minimize(cost)
    return op, cost, {
        'relative error': relative_error
    }


def make_batch(all_data, mask, size, timeslice_size):
    indices = np.random.choice(mask, size=(size, 1))
    indices = indices + np.arange(0, timeslice_size)
    return all_data[indices.astype(np.int32)]


def train(model):
    files = glob('./data/*.raw')
    all_data = np.zeros((0, 1), np.float32)
    for file in files:
        waveform = np.fromfile(files[0], dtype='<i2')
        waveform = np.reshape(waveform, (-1, 1))
        waveform = (waveform - np.mean(waveform)) / np.std(waveform)
        all_data = np.concatenate((all_data, waveform), axis=0)

    n_total = all_data.shape[0]
    n_train = 80 * n_total // 100
    batch_size = 4000
    batches = n_train // batch_size
    i_total = np.arange(0, n_total - model.slice_size, dtype=np.int32)
    shuffle(i_total)
    i_train = i_total[:n_train]
    i_test = i_total[n_train:]

    n_epochs = 200
    x = tf.placeholder(tf.float32, [None, model.slice_size, model.channels])
    op, cost, misc = optimiser(model, x)

    path = './save/{}'.format(model.__class__.__name__)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        try:
            saver.restore(session, path)
        except tf.errors.NotFoundError:
            session.run(init)
        for epoch in range(n_epochs):
            print('Epoch {}'.format(epoch))
            for i in range(batches):
                batch = make_batch(all_data, i_train, batch_size,
                                   model.slice_size)
                session.run(op, feed_dict={
                    x: batch
                })
                if i % 50 == 0:
                    train_error = session.run(cost, feed_dict={
                        x: batch
                    })
                    test_error, relative_error_value = session.run(
                        [cost, misc['relative error']],
                        feed_dict={
                            x: make_batch(all_data, i_test, batch_size,
                                          model.slice_size)
                        })
                    info = '{2:3d} / {3:3d} :' \
                           ' {0:2.2g} {1:2.2g} {4:2.2g}'.format(
                        train_error,
                        test_error, i,
                        batches,
                        relative_error_value
                    )
                    print(info)
                    saver.save(session, path)


if __name__ == '__main__':
    train(LinearModel(1225, 1225 // 4))
