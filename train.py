from random import shuffle

import tensorflow as tf
import numpy as np
from tensorflow.contrib import ffmpeg

from model import model, timeslice_size, samples_per_second

x = tf.placeholder(tf.float32, shape=(None, timeslice_size))
encoded, decoded = model(x)

mse = tf.reduce_mean(tf.square(x - decoded))
# cost = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=decoder(
#         encoder(x))))
optimiser = tf.train.AdamOptimizer()
op = optimiser.minimize(mse)

init = tf.global_variables_initializer()

file = tf.read_file(r'./bensound-goinghigher.mp3')
waveform_op = ffmpeg.decode_audio(file, file_format='mp3',
                                  samples_per_second=samples_per_second,
                                  channel_count=1)


def make_batch(all_data, mask, size):
    indices = np.random.choice(mask, size=(size, 1))
    indices = indices + np.arange(0, timeslice_size)
    return all_data[indices.astype(np.int32)]


with tf.Session() as session:
    waveform = session.run(waveform_op)
    waveform = waveform.reshape(-1)

n_total = waveform.shape[0]
n_train = 80 * n_total // 100
batch_size = 4000
batches = n_train // batch_size
i_total = np.arange(0, n_total - timeslice_size, dtype=np.int32)
shuffle(i_total)
i_train = i_total[:n_train]
i_test = i_total[n_train:]

print(np.mean(waveform[i_train]), np.std(waveform[i_train]))

# waveform = tf.constant(np.random.randn(n_total))
n_epochs = 20

saver = tf.train.Saver()
with tf.Session() as session:
    try:
        saver.restore(session, './save/audio-autoencoder')
    except tf.errors.NotFoundError:
        session.run(init)
    for epoch in range(n_epochs):
        print('Epoch {}'.format(epoch))
        for i in range(batches):
            batch = make_batch(waveform, i_train, batch_size)
            session.run(op, feed_dict={
                x: batch
            })
            if i % 50 == 0:
                train_error = session.run(mse, feed_dict={
                    x: batch
                })
                test_error = session.run(mse, feed_dict={
                    x: make_batch(waveform, i_test, batch_size)
                })
                print('{2:3d} / {3:3d} : {0:2.2g} {1:2.2g}'.format(
                    train_error,
                    test_error,
                    i,
                    batches
                ))
                saver.save(session, './save/audio-autoencoder')
