import datetime
import tensorflow as tf
import numpy as np

from .model import model
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


def main():
    from config import config_dict

    model_dir = config_dict['data']['logs'] \
                + f'/synth/{datetime.datetime.utcnow():%Y_%m_%d_%H_%M}'

    model_fn = model
    # model_fn = baseline_model(128, 256, model_dir)

    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=model_dir
    )

    (features_train, y_train), (features_test, y_test) = load_data(
        config_dict['data']['cache'])

    epochs = 200
    batch_size = 1000
    print(f'steps: {y_train.shape[0] * epochs // batch_size}')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        features_train,
        y_train,
        shuffle=True,
        batch_size=batch_size,
        num_epochs=epochs,
    )

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        features_test,
        y_test,
        shuffle=True,
        batch_size=batch_size,
        num_epochs=1,
    )

    train_and_test(
        estimator,
        20000,
        1000,
        train_input_fn,
        test_input_fn,
    )


if __name__ == '__main__':
    main()
