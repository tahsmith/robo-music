import sys

import numpy as np
import tensorflow as tf

from audio import ffmpeg
from synth.prepare import compute_features, clip_to_slice_size, \
    normalise_waveform, quantise, mu_law_decode
from synth.model import model_fn, params_from_config, model_width
from utils import normalise_to_int_range


def main(argv):
    model_path = argv[1]
    conditioning_file_path = argv[2]
    output_path = argv[3]
    try:
        time = int(argv[4])
    except IndexError:
        time = 2.0

    from config import config_dict
    sample_rate = config_dict['audio']['sample_rate']
    channels = config_dict['audio']['channels']
    slice_size = config_dict['synth']['feature_window']
    quantisation = config_dict['synth']['quantisation']
    depth = config_dict['synth']['dilation_stack_depth']
    count = config_dict['synth']['dilation_stack_count']
    n_mels = 128

    regenerate(model_path, conditioning_file_path, output_path, n_mels,
               channels, quantisation, sample_rate, slice_size, time, depth,
               count)


def regenerate(model_path, conditioning_file_path, output_path, n_mels,
               channels, quantisation, sample_rate, slice_size, time, depth,
               count):
    waveform, conditioning = load_conditioning(channels, conditioning_file_path,
                                               n_mels, sample_rate, slice_size,
                                               quantisation, time)

    waveform = regenerate_with_conditioning(model_path, waveform, quantisation,
                                            conditioning,
                                            model_width(depth, count) + 100)

    print(conditioning.shape[0])
    print(waveform.shape[0])

    ffmpeg.write_sound_file(waveform, output_path, sample_rate)


def load_conditioning(channels, conditioning_file_path, n_mels, sample_rate,
                      slice_size, quantisation, time):
    conditioning_waveform = ffmpeg.load_sound_file(
        conditioning_file_path,
        dtype=np.int16,
        sample_rate=sample_rate,
        channels=channels
    ).astype(np.float32)

    conditioning_waveform = conditioning_waveform[:int(sample_rate * time), :]

    conditioning_waveform = clip_to_slice_size(slice_size,
                                               conditioning_waveform)

    conditioning_waveform = normalise_waveform(conditioning_waveform, channels)

    conditioning = compute_features(
        conditioning_waveform,
        sample_rate,
        slice_size,
        n_mels
    )

    conditioning_waveform = quantise(conditioning_waveform, quantisation)

    conditioning += np.random.randn(*conditioning.shape)

    assert conditioning.shape[0] == conditioning_waveform.shape[0]

    return conditioning_waveform, conditioning


def regenerate_with_conditioning(model_path, init_waveform, quantisation,
                                 conditioning, slice_size):
    waveform = tf.get_variable('waveform', shape=[slice_size - 1, 1],
                               dtype=tf.int32)
    conditioning_tf = tf.Variable(conditioning, dtype=tf.float32)
    limit = tf.get_variable('limit', shape=[], dtype=tf.int32)

    def cond(i, _):
        return tf.less(i, limit)

    def loop(i, waveform_):
        features = {
            'waveform': waveform_[tf.newaxis, -slice_size + 1:, :],
            'conditioning': conditioning_tf[i:i + 1, :]
        }

        estimator_spec = model_fn(features, tf.estimator.ModeKeys.PREDICT,
                                  params_from_config())

        next_point = estimator_spec.predictions

        waveform_ = tf.concat(
            [waveform_, next_point[0, -1:, tf.newaxis]],
            axis=0
        )

        return [tf.add(i, 1), waveform_]

    while_op = tf.while_loop(
        cond,
        loop,
        loop_vars=[0, waveform],
        shape_invariants=[tf.TensorShape(()), tf.TensorShape([None, 1])],
        parallel_iterations=1
    )

    sess = tf.Session()
    new_saver = tf.train.Saver(var_list=tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, 'synth'))
    sess.run(tf.global_variables_initializer())
    new_saver.restore(sess, tf.train.latest_checkpoint(model_path))

    output_waveform = init_waveform[:slice_size - 1, :]
    for i in range(conditioning.shape[0] // 1000):
        sess.run(limit.assign(1000))
        sess.run(waveform.assign(output_waveform[-(slice_size - 1):, :]))
        while_op_result = sess.run(while_op)
        final_i, waveform_result = while_op_result
        output_waveform = np.concatenate(
            (output_waveform, waveform_result[slice_size - 1:, :]))
        print(f'{i * 1000}')

    waveform_result = output_waveform[slice_size - 1:, :]
    waveform_result = mu_law_decode(waveform_result, quantisation)
    waveform_result = normalise_to_int_range(waveform_result, np.int16)
    print(np.max(waveform_result))
    print(np.min(waveform_result))
    print(np.mean(waveform_result))

    return waveform_result


if __name__ == '__main__':
    main(sys.argv)
