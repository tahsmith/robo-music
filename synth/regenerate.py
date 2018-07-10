import os
import sys

import numpy as np
import tensorflow as tf

from audio import ffmpeg
from synth.prepare import compute_features, clip_to_slice_size
from synth.model import model_fn, params_from_config
from utils import dilate_zero_order_hold


def main(argv):
    model_path = argv[1]
    conditioning_file_path = argv[2]
    output_path = argv[3]

    from config import config_dict
    sample_rate = config_dict['audio']['sample_rate']
    channels = config_dict['audio']['channels']
    slice_size = config_dict['synth']['slice_size']
    quantisation = config_dict['synth']['quantisation']
    n_mels = 128

    regenerate(model_path, conditioning_file_path, output_path, n_mels,
               channels, quantisation, sample_rate, slice_size)


def regenerate(model_path, conditioning_file_path, output_path, n_mels,
               channels, quantisation, sample_rate, slice_size):
    conditioning = load_conditioning(channels, conditioning_file_path,
                                     n_mels, sample_rate, slice_size)
    estimator = load_model(model_path)

    def loop_hook(i, total):
        if i % slice_size == 0:
            print(f'{i} / {total}')

    waveform = regenerate_with_conditioning(estimator, slice_size, quantisation,
                                            conditioning, loop_hook)
    ffmpeg.write_sound_file(waveform, output_path, sample_rate)


def load_conditioning(channels, conditioning_file_path, n_mels, sample_rate,
                      slice_size):
    conditioning_waveform = ffmpeg.load_sound_file(
        conditioning_file_path,
        dtype=np.int16,
        sample_rate=sample_rate,
        channels=channels
    ).astype(np.float32)

    conditioning_waveform = clip_to_slice_size(slice_size,
                                               conditioning_waveform)
    conditioning = compute_features(
        conditioning_waveform,
        sample_rate,
        slice_size,
        slice_size,
        n_mels
    )

    conditioning = dilate_zero_order_hold(conditioning, slice_size)

    return conditioning


def load_model(model_path):
    sess = tf.Session()
    tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        model_path
    )

    print(sess.graph.get_operations())
    waveform = sess.graph.get_tensor_by_name('waveform:0')
    conditioning = sess.graph.get_tensor_by_name('conditioning:0')
    predictions = sess.graph.get_tensor_by_name('prediction:0')

    def predict_fn(wave_slice, conditioning_slice):
        sess.run(
            [predictions],
            feed_dict={
                waveform: wave_slice,
                conditioning: conditioning_slice
            }
        )

    return predict_fn


def regenerate_with_conditioning(predict_fn, slice_size, quantisation,
                                 conditioning,
                                 loop_hook):
    waveform = np.random.randint(
        0, 1, (conditioning.shape[0], 1), dtype=np.int32) * \
               (quantisation // 2)
    waveform[slice_size - 2] = np.random.randint(0, quantisation - 1)

    total = conditioning.shape[0] - slice_size + 1
    for i in range(0, total):
        begin = i
        end = begin + slice_size - 1

        conditioning_slice = conditioning[begin:begin + 1, :]
        wave_slice = np.array([waveform[begin:end]])

        prediction = predict_fn(wave_slice, conditioning_slice)
        waveform[i + slice_size] = prediction

        loop_hook(i + 1, total)

    return waveform.astype(np.int16)


if __name__ == '__main__':
    main(sys.argv)
