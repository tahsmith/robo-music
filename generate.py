import tensorflow as tf
import numpy as np
from tensorflow.contrib import ffmpeg

from model import model, samples_per_second, timeslice_size

n = 10 * (samples_per_second // timeslice_size)
random_encoded = tf.constant(np.ones((n, 50)) + 0.001 * np.random.randn(n, 50), dtype=tf.float32)
encoder, decoder = model()
slices_output = decoder(random_encoded)
audio_output = tf.reshape(slices_output, (-1, 1))

out_format = ffmpeg.encode_audio(audio_output, file_format='mp3',
                                 samples_per_second=samples_per_second)
output_file = tf.write_file('output.mpf', out_format)
saver = tf.train.Saver()

with tf.Session() as session:
    saver.restore(session, './save/audio-autoencoder')
    session.run(output_file)
