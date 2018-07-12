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

    filter_ = add_conditioning(filter_, conditioning_inputs)
    gate = add_conditioning(gate, conditioning_inputs)

    return tf.sigmoid(gate) * tf.tanh(filter_)


def conv(inputs):
    return tf.layers.conv1d(inputs,
                            filters=32,
                            kernel_size=2,
                            strides=2,
                            padding='valid')


def params_from_config():
    from config import config_dict
    return {
        'quantisation': config_dict['synth']['quantisation']
    }


def model_fn(features, labels, mode, params):
    with tf.variable_scope('synth'):
        waveform = features['waveform']
        conditioning = features['conditioning']
        quantisation = params['quantisation']

        encoded = tf.one_hot(
            waveform,
            quantisation
        )
        encoded = tf.reshape(encoded, [-1, 2047, quantisation])

        output = tf.layers.conv1d(encoded, kernel_size=2, strides=1, filters=32)

        for i in range(5):
            output = layer(output, conv, conditioning, mode)

        flatten = tf.layers.flatten(output)
        dropout = tf.layers.dropout(flatten,
                                    training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(dropout, quantisation)
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode, predictions,
                export_outputs={'predict_output': tf.estimator.export.PredictOutput(
                    {"predictions": predictions,
                     'probabilities': tf.nn.softmax(logits)})})

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)

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
