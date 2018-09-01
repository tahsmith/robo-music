import datetime
import glob

import sys
from functools import partial

import tensorflow as tf
import numpy as np

from synth.model import params_from_config, ModelParams
from .model import model_fn
from train_utils import train_and_test


def load_batch(file):
    waveform_test = np.load(file)
    assert waveform_test.dtype == np.int32
    x_test = waveform_test[:, 0:-1]
    y_test = waveform_test[:, -1]
    return x_test, y_test


def baseline_model(conditioning_features, quantisation, model_dir):
    return tf.estimator.LinearClassifier(
        [
            tf.feature_column.categorical_column_with_identity('waveform',
                                                               quantisation),
            tf.feature_column.numeric_column('conditioning',
                                             (conditioning_features,))
        ],
        model_dir=model_dir,
        n_classes=quantisation
    ).model_fn


def augment_sample(sample, noise_level=0.0, scale_range=1.0):
    scale = scale_range ** tf.random_uniform((tf.shape(sample)[0], 1, 1), -1.0,
                                             1.0)
    noise = tf.random_normal(tf.shape(sample), 0.0, noise_level)

    sample = scale * sample
    sample = noise + sample

    return sample


def normalise_waveform(waveform):
    waveform = waveform - tf.reduce_mean(waveform)
    waveform_range = tf.reduce_max(tf.abs(waveform))
    waveform = waveform / waveform_range

    return waveform


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    mu = quantization_channels - 1.0
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    mu_law = tf.sign(audio) * tf.log1p(mu * np.abs(audio)) / np.log1p(mu)
    return tf.cast((mu_law + 1) / 2 * mu + 0.5, tf.int32)


def quantise(x, quantisation):
    if quantisation != 256:
        raise NotImplementedError('quantisation != 256')
    return mu_law_encode(x, quantisation)


def random_slices(waveform, feature, slice_size,
                  params: ModelParams):
    max_index = waveform.shape[0] - slice_size
    channels = waveform.shape[1]
    while True:
        start = int(np.random.uniform(0, max_index))
        end = start + slice_size

        waveform_batch = waveform[start:end, :]

        feature_batch = feature[start:end, :]

        yield {
            'waveform': waveform_batch,
            'conditioning': feature_batch
        }


def normalise_and_augment(data_point, params):
    return {
        'waveform': quantise(
            normalise_waveform(
                augment_sample(
                    data_point['waveform'],
                    0.01,
                    1.2
                )
            ),
            params.quantisation
        ),
        'conditioning': data_point['conditioning']
    }


def input_function_from_array(waveform, feature, params, slice_size,
                              batch_size):
    def input_fn():
        return tf.data.Dataset.from_generator(
            partial(random_slices, waveform, feature, slice_size, params),
            {'waveform': tf.float32, 'conditioning': tf.float32},
            {
                'waveform': tf.TensorShape((slice_size, params.channels)),
                'conditioning': tf.TensorShape((slice_size, params.n_mels))
            }
        ).batch(batch_size) \
            .map(partial(normalise_and_augment, params=params)) \
            .prefetch(1)

    return input_fn


def main(argv):
    from config import config_dict

    data_config = config_dict['data']
    synth_config = config_dict['synth']

    try:
        model_dir = argv[1]
    except IndexError:
        model_dir = data_config['logs'] \
                    + f'/synth/{datetime.datetime.utcnow():%Y_%m_%d_%H_%M}'

    # model_fn = baseline_model(128, 256, model_dir)

    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=model_dir,
        params=params_from_config()
    )

    params = params_from_config()

    train_waveform = np.load(
        f"{config_dict['data']['cache']}/synth/waveform_train.npy"
    )

    train_features = np.load(
        f"{config_dict['data']['cache']}/synth/features_train.npy"
    )

    test_waveform = np.load(
        f"{config_dict['data']['cache']}/synth/waveform_test.npy"
    )

    test_features = np.load(
        f"{config_dict['data']['cache']}/synth/features_test.npy"
    )

    batch_size = synth_config['batch_size']
    steps_per_evals = synth_config['steps_per_eval']
    evals_steps = synth_config['eval_steps']

    train_input_fn = input_function_from_array(
        train_waveform,
        train_features,
        params,
        synth_config['slice_size'],
        batch_size
    )

    test_input_fn = input_function_from_array(
        test_waveform,
        test_features,
        params,
        synth_config['slice_size'],
        batch_size
    )

    train_and_test(estimator, train_input_fn, test_input_fn,
                   synth_config['steps'],
                   steps_per_evals,
                   evals_steps)


if __name__ == '__main__':
    main(sys.argv)
