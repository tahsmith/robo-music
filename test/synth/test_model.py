from synth.model import ModelParams, model_fn, model_width
import tensorflow as tf
import numpy as np


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
