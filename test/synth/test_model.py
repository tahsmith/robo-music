import numpy as np
import pytest
import tensorflow as tf

from synth.model import ModelParams, model_fn, model_width
from synth.prepare import mu_law_encode, mu_law_decode, compute_features


@pytest.fixture
def sess():
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        yield sess


@pytest.fixture
def params():
    params = ModelParams(
        channels=1,
        dilation_stack_depth=2,
        dilation_stack_count=2,
        residual_filters=8,
        conv_filters=8,
        skip_filters=16,
        quantisation=256,
        regularisation=False,
        dropout=False,
        conditioning=False,
        sample_rate=11025,
        feature_window=2048,
        n_mels=128
    )

    return params


def make_inputs(params, slice_size):
    return {
        'waveform': tf.constant(
            np.zeros((2, slice_size, params.channels)),
            tf.int32
        ),
        'conditioning': tf.constant(
            np.zeros((2, slice_size, params.n_mels)),
            tf.float32
        )
    }


def test_model_shape_train(sess, params):
    features = make_inputs(params, params.receptive_field + 1)
    model = model_fn(
        features,
        tf.estimator.ModeKeys.TRAIN,
        params
    )

    sess.run(tf.global_variables_initializer())
    predictions = sess.run(model.predictions)
    assert predictions.shape[1] == 1


def test_model_shape_eval(sess, params):
    features = make_inputs(params, params.receptive_field + 1)
    model = model_fn(
        features,
        tf.estimator.ModeKeys.EVAL,
        params
    )

    sess.run(tf.global_variables_initializer())
    predictions = sess.run(model.predictions)
    assert predictions.shape[1] == 1


def test_model_shape_predict(sess, params):
    features = make_inputs(params, params.receptive_field)
    model = model_fn(
        features,
        tf.estimator.ModeKeys.PREDICT,
        params
    )

    sess.run(tf.global_variables_initializer())
    predictions = sess.run(model.predictions)
    assert predictions.shape[1] == 1


def test_conditioning_shape_train(sess, params):
    params.conditioning = True
    features = make_inputs(params, params.receptive_field + 1)
    model = model_fn(
        features,
        tf.estimator.ModeKeys.TRAIN,
        params
    )

    sess.run(tf.global_variables_initializer())
    predictions = sess.run(model.predictions)
    assert predictions.shape[1] == 1


def test_conditioning_shape_eval(sess, params):
    params.conditioning = True
    features = make_inputs(params, params.receptive_field + 1)
    model = model_fn(
        features,
        tf.estimator.ModeKeys.EVAL,
        params
    )

    sess.run(tf.global_variables_initializer())
    predictions = sess.run(model.predictions)
    assert predictions.shape[1] == 1


def test_conditioning_shape_predict(sess, params):
    params.conditioning = True
    features = make_inputs(params, params.receptive_field)
    model = model_fn(
        features,
        tf.estimator.ModeKeys.PREDICT,
        params
    )

    sess.run(tf.global_variables_initializer())
    predictions = sess.run(model.predictions)
    assert predictions.shape[1] == 1


@pytest.mark.skipif('not tf.test.is_gpu_available(cuda_only=True)')
def test_model_train(sess, params):
    sample_rate = 44100
    time_length = 1
    n_points = sample_rate * time_length
    params.slice_size = n_points

    t = np.arange(0, n_points) / sample_rate
    freq = 440
    sine_wave = np.sin(t * 2 * np.pi * t * freq).reshape((-1, 1))
    conditioning = compute_features(sine_wave, sample_rate,
                                    params.feature_window, params.n_mels)
    sine_wave = sine_wave[np.newaxis, :, :]
    conditioning = conditioning[np.newaxis, :, :]

    sine_wave_encoded = mu_law_encode(sine_wave, params.quantisation)

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
