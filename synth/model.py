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

    # filter_ = add_conditioning(filter_, conditioning_inputs)
    # gate = add_conditioning(gate, conditioning_inputs)

    return tf.sigmoid(gate) * tf.tanh(filter_)


def conv(inputs):
    return tf.layers.conv1d(inputs,
                            filters=10,
                            kernel_size=2,
                            strides=2,
                            padding='valid')


def model(features, labels, mode):
    waveform = features['waveform']
    conditioning = features['conditioning']

    output = waveform

    for i in range(5):
        output = layer(output, conv, conditioning, mode)

    flatten = tf.layers.flatten(output)
    dropout = tf.layers.dropout(flatten,
                                training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(dropout, 256)
    predictions = tf.argmax(logits, axis=1)

    loss = tf.reduce_sum(tf.square(output - labels))

    training_op = tf.train.AdamOptimizer().minimize(loss,
                                                    tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode,
            output,
            loss,
            training_op
        )

    return tf.estimator.EstimatorSpec(
        mode,
        output,
        loss,
        tf.metrics.accuracy(labels, output)
    )
