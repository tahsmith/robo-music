import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import ffmpeg

from model import model, samples_per_second, timeslice_size

file_name = sys.argv[1]
input_file = tf.read_file(file_name)
input_data = ffmpeg.decode_audio(input_file,
                                 file_format=file_name.split('.')[-1],
                                 samples_per_second=samples_per_second,
                                 channel_count=1)
input_len = tf.shape(input_data)[0]
input_data = tf.reshape(
    tf.slice(input_data, [0, 0], [input_len - input_len % timeslice_size, 1]),
    [-1, timeslice_size])

with tf.Session() as session:
    batches = session.run(tf.shape(input_data))[0]

encoded, decoded = model(input_data, timeslice_size, 1, batches)
slices_output = decoded
output_data = tf.reshape(slices_output, (-1, 1))
output_wav = ffmpeg.encode_audio(output_data, file_format='wav',
                                 samples_per_second=samples_per_second)
output_file = tf.write_file(r'./regenerated.wav', output_wav)
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as session:
    saver.restore(session, './save/audio-autoencoder')
    # session.run(init)
    print("Size ", session.run([
        output_file, tf.size(output_data)
    ]))
