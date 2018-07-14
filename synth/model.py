from functools import partial

import tensorflow as tf


def add_conditioning(inputs, conditioning):
    conditioning_layer = tf.layers.dense(conditioning, 1)
    shape = tf.shape(inputs)

    input_flattened = tf.reshape(inputs, [shape[0], shape[1] * shape[2]])

    output = conditioning_layer + input_flattened
    output = tf.reshape(output, shape)

    return output


def layer(inputs, conv_fn, conditioning_inputs, mode):
    filter_ = conv_fn(inputs)
    gate = conv_fn(inputs)

    if conditioning_inputs is not None:
        filter_ = add_conditioning(filter_, conditioning_inputs)
        gate = add_conditioning(gate, conditioning_inputs)

    return tf.sigmoid(gate) * tf.tanh(filter_)


def conv1d(inputs, filters):
    return tf.layers.conv1d(inputs,
                            filters=filters,
                            kernel_size=2,
                            strides=2,
                            padding='valid')


def params_from_config():
    from config import config_dict
    synth_config = config_dict['synth']
    return {
        'layers': synth_config['layers'],
        'filters': synth_config['filters'],
        'quantisation': synth_config['quantisation'],
        'regularisation': synth_config['regularisation'],
        'dropout': synth_config['dropout'],
        'conditioning': synth_config['conditioning']
    }


def model_fn(features, labels, mode, params):
    with tf.variable_scope('synth'):
        waveform = features['waveform']
        if params['conditioning']:
            conditioning = features['conditioning']
        else:
            conditioning = None

        filters = params['filters']
        quantisation = params['quantisation']
        regularisation = params['regularisation']
        dropout = params['dropout']
        layers = params['layers']

        encoded = tf.one_hot(
            waveform,
            quantisation
        )
        encoded = tf.reshape(encoded, [-1, 2047, quantisation])

        output = tf.layers.conv1d(encoded, kernel_size=2, strides=1,
                                  filters=filters)

        conv = partial(conv1d, filters=filters)

        for i in range(layers):
            output = layer(output, conv, conditioning, mode)
            if dropout:
                output = tf.layers.dropout(
                    output,
                    rate=dropout,
                    training=mode == tf.estimator.ModeKeys.TRAIN
                )

        flatten = tf.layers.flatten(output)
        logits = tf.layers.dense(flatten, quantisation)
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode, predictions,
                export_outputs={
                    'predict_output': tf.estimator.export.PredictOutput(
                        {"predictions": predictions,
                         'probabilities': tf.nn.softmax(logits)})})

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)

        if regularisation:
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               'synth')

            reg_loss = tf.add_n([tf.nn.l2_loss(x) for x in trainable_vars])
            loss += regularisation * reg_loss

        training_op = tf.train.AdamOptimizer().minimize(loss,
                                                        tf.train.get_global_step())

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                mode,
                predictions,
                loss,
                training_op
            )

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predictions)
        }

        return tf.estimator.EstimatorSpec(
            mode,
            predictions,
            loss,
            eval_metric_ops=eval_metric_ops
        )
