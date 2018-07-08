import datetime
import tensorflow as tf
import numpy as np

from utils import normalise_to_int_range

from .model import model


def load_batch(file):
    waveform_test = np.load(file)
    assert waveform_test.dtype == np.int32
    x_test = waveform_test[:, 0:-1]
    y_test = waveform_test[:, -1]
    return x_test, y_test


def load_data(cache_path):
    x_test, y_test = load_batch(f'{cache_path}/synth/waveform_test.npy')
    conditioning_test = np.load(f'{cache_path}/synth/features_test.npy')

    waveform_train = np.load(f'{cache_path}/synth/waveform_train.npy')
    x_train = waveform_train[:, 0:-1]
    y_train = waveform_train[:, -1]
    conditioning_train = np.load(f'{cache_path}/synth/features_train.npy')

    features_test = {
        'waveform':  x_test,
        'conditioning': conditioning_test
    }
    
    features_train = {
        'waveform':  x_train,
        'conditioning': conditioning_train
    }

    return (features_train, y_train), (features_test, y_test)


def main():
    from config import config_dict

    model_fn = model

    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=config_dict['data']['logs']
                  + f'/synth/{datetime.datetime.utcnow():%Y_%m_%d_%H_%M}'
    )
    
    (features_train, y_train), (features_test, y_test) = load_data(
        config_dict['data']['cache'])

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        features_train,
        y_train,
        shuffle=True,
        batch_size=1000,
        num_epochs=2000,
    )

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        features_test,
        y_test,
        shuffle=True,
        batch_size=1000,
        num_epochs=1,
    )

    estimator.train(
        input_fn=train_input_fn,
        steps=None,
    )
    estimator.evaluate(input_fn=test_input_fn, steps=None)


if __name__ == '__main__':
    main()
