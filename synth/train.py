import datetime
import glob

import sys
from functools import partial
from random import shuffle

import tensorflow as tf
import numpy as np

from synth.model import params_from_config
from .model import model_fn
from train_utils import train_and_test


def load_batch(file):
    waveform_test = np.load(file)
    assert waveform_test.dtype == np.int32
    x_test = waveform_test[:, 0:-1]
    y_test = waveform_test[:, -1]
    return x_test, y_test


def load_data(cache_path):
    x_train, y_train = load_batch(f'{cache_path}/synth/waveform_train.npy')
    conditioning_train = np.load(f'{cache_path}/synth/features_train.npy')

    x_test, y_test = load_batch(f'{cache_path}/synth/waveform_test.npy')
    conditioning_test = np.load(f'{cache_path}/synth/features_test.npy')

    features_train = {
        'waveform': x_train,
        'conditioning': conditioning_train
    }

    features_test = {
        'waveform': x_test,
        'conditioning': conditioning_test
    }

    return (features_train, y_train), (features_test, y_test)


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


def input_generator(waveform_files, feature_files, batch_size):
    pairs = list(zip(waveform_files, feature_files))
    shuffle(pairs)
    for waveform_file, feature_file in pairs:
        waveform, label = load_batch(waveform_file)
        features = np.load(feature_file)
        for i in range((waveform.shape[0] // batch_size) - 1):
            begin = i * batch_size
            end = begin + batch_size
            yield (
                {
                    'waveform': waveform[begin:end, :, :],
                    'conditioning': features[begin:end, :]
                },
                label[begin:end]
            )


def input_function_from_file(waveform_files, feature_files, batch_size,
                             prefetch):
    def input_fn():
        return tf.data.Dataset.from_generator(
            partial(input_generator, waveform_files, feature_files, batch_size),
            ({'waveform': tf.int32, 'conditioning': tf.float32}, tf.int32)
        ).prefetch(prefetch)

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

    batch_size = synth_config['batch_size']

    waveform_files = glob.glob(
        f"{config_dict['data']['cache']}/synth/waveform*.npy")

    feature_files = glob.glob(
        f"{config_dict['data']['cache']}/synth/features*.npy")

    n_files = len(waveform_files)
    train_cut = 90 * n_files // 100
    steps_per_evals = synth_config['steps_per_eval']
    evals_steps = synth_config['eval_steps']

    train_input_fn = input_function_from_file(
        waveform_files[:train_cut],
        feature_files[:train_cut],
        batch_size,
        steps_per_evals
    )

    test_input_fn = input_function_from_file(
        waveform_files[train_cut:],
        feature_files[train_cut:],
        batch_size,
        evals_steps
    )

    train_and_test(estimator, train_input_fn, test_input_fn,
                   synth_config['steps'],
                   steps_per_evals,
                   evals_steps)


if __name__ == '__main__':
    main(sys.argv)
