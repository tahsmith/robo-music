import numpy as np
import pytest
import tensorflow as tf

from synth.model import ModelParams, model_fn, model_width
from synth.prepare import mu_law_encode


def make_test_inputs(slice_size, channel_size, batch_size,
                     conditioning_features):
    return {
        'waveform': tf.constant(
            np.zeros((batch_size, slice_size, channel_size)),
            tf.int32
        ),
        'conditioning': tf.constant(
            np.zeros((batch_size, slice_size, conditioning_features)),
            tf.float32
        )
    }


def test_model_width():
    params = ModelParams(
        slice_size=1,
        channels=1,
        dilation_stack_depth=10,
        dilation_stack_count=5,
        residual_filters=32,
        conv_filters=32,
        skip_filters=512,
        quantisation=256,
        regularisation=False,
        dropout=False,
        conditioning=False
    )

    params.slice_size = model_width(params.dilation_stack_depth,
                                    params.dilation_stack_count)
    print(params.slice_size)

    features = make_test_inputs(params.slice_size, params.channels, 2, 1)
    model = model_fn(
        features,
        tf.estimator.ModeKeys.PREDICT,
        params
    )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        predictions = sess.run(model.predictions)
        assert predictions.shape[1] == 1


@pytest.mark.skipif('not tf.test.is_gpu_available(cuda_only=True)')
def test_model_train():
    sample_rate = 44100
    time_length = 1
    n_points = sample_rate * time_length
    t = np.arange(0, n_points) / sample_rate
    freq = 440
    sine_wave = np.sin(t * 2 * np.pi * t * freq)
    sine_wave = sine_wave[np.newaxis, :, np.newaxis]
    sine_wave = mu_law_encode(sine_wave, 256)
    conditioning = np.random.randn(1, n_points, 256)

    params = ModelParams(
        slice_size=n_points,
        channels=1,
        dilation_stack_depth=8,
        dilation_stack_count=2,
        residual_filters=8,
        conv_filters=16,
        skip_filters=32,
        quantisation=256,
        regularisation=False,
        dropout=False,
        conditioning=False
    )

    print(params.receptive_field)

    train_spec = model_fn(
        {'waveform': tf.constant(sine_wave),
         'conditioning': tf.constant(conditioning)},
        tf.estimator.ModeKeys.TRAIN,
        params
    )

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(1000):
            train_value, loss_value = session.run([train_spec.train_op,
                                                   train_spec.loss])
            print(f'{loss_value}')
            if loss_value < 0.8:
                break
        else:
            raise AssertionError(
                f'Training did not converge. Final loss: {loss_value}'
            )
