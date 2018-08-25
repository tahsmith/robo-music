import numpy as np
import pytest
import tensorflow as tf

from synth.model import ModelParams, model_fn, model_width
from synth.prepare import mu_law_encode, mu_law_decode


@pytest.fixture
def sess():
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        yield sess


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


def test_model_width(sess):
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
        conditioning=False,
        sample_rate=11025,
        feature_window=2048,
        n_mels=128
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

    sess.run(tf.global_variables_initializer())
    predictions = sess.run(model.predictions)
    assert predictions.shape[1] == 1


def test_conditioning(sess):
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
        conditioning=True,
        sample_rate=11025,
        feature_window=2048,
        n_mels=128
    )

    params.slice_size = model_width(params.dilation_stack_depth,
                                    params.dilation_stack_count)

    features = make_test_inputs(params.slice_size, params.channels, 2,
                                params.n_mels)
    model = model_fn(
        features,
        tf.estimator.ModeKeys.TRAIN,
        params
    )

    sess.run(tf.global_variables_initializer())
    predictions = sess.run(model.predictions)
    assert predictions.shape[1] == 1


@pytest.mark.skipif('not tf.test.is_gpu_available(cuda_only=True)')
def test_model_train(sess):
    sample_rate = 44100
    time_length = 1
    n_points = sample_rate * time_length

    params = ModelParams(
        slice_size=n_points,
        channels=1,
        dilation_stack_depth=8,
        dilation_stack_count=2,
        residual_filters=8,
        conv_filters=8,
        skip_filters=8,
        quantisation=256,
        regularisation=False,
        dropout=False,
        conditioning=False,
        sample_rate=11025,
        feature_window=2048,
        n_mels=128
    )

    t = np.arange(0, n_points) / sample_rate
    freq = 440
    sine_wave = np.sin(t * 2 * np.pi * t * freq)
    sine_wave = sine_wave[np.newaxis, :, np.newaxis]
    sine_wave_encoded = mu_law_encode(sine_wave, params.quantisation)
    conditioning = np.random.randn(1, n_points, 1)

    print(params.receptive_field)

    train_spec = model_fn(
        {'waveform': tf.constant(sine_wave_encoded),
         'conditioning': tf.constant(conditioning)},
        tf.estimator.ModeKeys.TRAIN,
        params
    )

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        predict_spec = model_fn(
            {'waveform': tf.constant(sine_wave_encoded),
             'conditioning': tf.constant(conditioning)},
            tf.estimator.ModeKeys.PREDICT,
            params
        )

    sess.run(tf.global_variables_initializer())

    for i in range(2000):
        train_value, loss_value = sess.run([train_spec.train_op,
                                            train_spec.loss])
        print(f'{i} - {loss_value}')
        if loss_value < 0.8:
            break
    else:
        raise AssertionError(
            f'Training did not converge. Final loss: {loss_value}'
        )

    predictions = sess.run(predict_spec.predictions)
    actual = mu_law_decode(predictions[0, :-1], params.quantisation)
    expected = sine_wave[0, params.receptive_field:, 0]
    accuracy = np.mean(np.abs(actual - expected))
    # Channels are quantised as per mu_law_encode, assert that average error
    # is less that the quantisation level, i.e, 1.
    assert accuracy < 1.0
