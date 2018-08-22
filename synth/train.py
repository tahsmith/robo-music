import datetime
import glob

import sys
from functools import partial
from random import shuffle

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
    sample = sample.astype(np.float32)
    scale = scale_range ** np.random.uniform(-1.0, 1.0)
    noise = np.random.randn(*sample.shape) * noise_level

    sample *= scale
    sample += noise

    return sample


def input_generator(waveform, feature, batch_size, params: ModelParams):
    max_index = waveform.shape[0] - params.slice_size
    channels = waveform.shape[1]
    n_features = feature.shape[1]
    while True:
        waveform_batch = np.empty((params.slice_size, channels))
        feature_batch = np.empty((params.slice_size, n_features))
        for i in range(batch_size):
            start = np.random.uniform(0, max_index)
            end = start + params.slice_size
            waveform_batch[i, :] = augment_sample(
                waveform[start:end, :],
                0.01,
                1.2
            )
            feature_batch[i, :] = feature[start:end, :]

        yield {
            'waveform': waveform_batch,
            'feature': feature_batch
        }


def input_function_from_array(waveform, feature, slice_size, batch_size,
                              prefetch):
    def input_fn():
        return tf.data.Dataset.from_generator(
            partial(input_generator, waveform, feature, slice_size, batch_size),
            {'waveform': tf.int32, 'conditioning': tf.float32}
        ).prefetch(prefetch).repeat()

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
        params.slice_size,
        batch_size,
        steps_per_evals,
    )

    test_input_fn = input_function_from_array(
        test_waveform,
        test_features,
        params.slice_size,
        batch_size,
        steps_per_evals,
    )

    train_and_test(estimator, train_input_fn, test_input_fn,
                   synth_config['steps'],
                   steps_per_evals,
                   evals_steps)


if __name__ == '__main__':
    main(sys.argv)
