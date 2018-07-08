import tensorflow as tf
import numpy as np
from tensorflow.contrib import ffmpeg

from synth.model import model, samples_per_second, timeslice_size

n = 10 * (samples_per_second // timeslice_size)
random_encoded = tf.constant(
    np.ones((n, timeslice_size, 1))
    + 0.001 * np.random.randn(n, timeslice_size, 1),
    dtype=tf.float32)
encoder, decoder = model(random_encoded, timeslice_size, 1)
slices_output = decoder
audio_output = tf.reshape(slices_output, (-1, 1))

out_format = ffmpeg.encode_audio(audio_output, file_format='wav',
                                 samples_per_second=samples_per_second)
output_file = tf.write_file('output.wav', out_format)
saver = tf.train.Saver()

with tf.Session() as session:
    saver.restore(session, './save/audio-synth')
    session.run(output_file)