import sys
import tensorflow as tf
from tensorflow.contrib import ffmpeg

from models import LinearModel, ConvModel
from prepare import slice_size

samples_per_second = 44100


def regenerate(model, file_name):
    input_file = tf.read_file(file_name)
    input_data = ffmpeg.decode_audio(input_file,
                                     file_format=file_name.split('.')[-1],
                                     samples_per_second=samples_per_second,
                                     channel_count=1)
    input_len = tf.shape(input_data)[0]
    input_data = tf.reshape(
        tf.slice(
            input_data,
            [0, 0],
            [input_len - input_len % model.slice_size, 1]
        ),
        [-1, model.slice_size, 1]
    )

    decoded = model.reconstructed(model.prepare(input_data))
    slices_output = decoded
    output_data = tf.reshape(slices_output, (-1, 1))
    output_wav = ffmpeg.encode_audio(output_data,
                                     file_format='wav',
                                     samples_per_second=samples_per_second)
    output_file = tf.write_file(r'./regenerated.wav', output_wav)
    saver = tf.train.Saver()

    with tf.Session() as session:
        saver.restore(session, sys.argv[2])
        session.run(
            output_file
        )


if __name__ == '__main__':
    regenerate(LinearModel(slice_size, slice_size // 2), sys.argv[1])
