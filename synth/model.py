import tensorflow as tf


def add_conditioning(inputs, conditioning):
    conditioning_layer = tf.layers.dense(conditioning, 1)
    shape = tf.shape(inputs)

    input_flattened = tf.reshape(inputs, [shape[0], shape[1] * shape[2]])

    output = conditioning_layer + input_flattened
    output = tf.reshape(output, shape)

    return output


def layer(inputs, conditioning_inputs, filters, dilation, mode):
    with tf.name_scope('conv_layer'):
        with tf.name_scope('filter'):
            filter_ = conv1d(inputs, filters, dilation)
            if conditioning_inputs is not None:
                filter_ = add_conditioning(filter_, conditioning_inputs)

            filter_ = tf.tanh(filter_)

        with tf.name_scope('gate'):
            gate = conv1d(inputs, filters, dilation)
            if conditioning_inputs is not None:
                gate = add_conditioning(gate, conditioning_inputs)

            gate = tf.sigmoid(gate)

        return filter_ * gate + inputs[:, dilation:, :]


def conv1d(inputs, filters, dilation):
    return tf.layers.conv1d(inputs,
                            filters=filters,
                            kernel_size=2,
                            dilation_rate=dilation,
                            padding='valid')


def params_from_config():
    from config import config_dict
    audio_config = config_dict['audio']
    synth_config = config_dict['synth']
    return {
        'channels': audio_config['channels'],
        'dilation_stack_depth': synth_config['dilation_stack_depth'],
        'dilation_stack_count': synth_config['dilation_stack_count'],
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

        channels = params['channels']
        filters = params['filters']
        quantisation = params['quantisation']
        regularisation = params['regularisation']
        dropout = params['dropout']
        dilation_stack_depth = params['dilation_stack_depth']
        dilation_stack_count = params['dilation_stack_count']

        assert channels == 1
        one_hot = tf.one_hot(
            waveform[:, :, 0],
            quantisation
        )

        one_hot = tf.reshape(one_hot, [-1, 2047, quantisation])

        output = tf.layers.conv1d(one_hot, kernel_size=2, strides=1,
                                  filters=filters)

        dilation_layers = [
            2 ** i
            for _ in range(dilation_stack_count)
            for i in range(dilation_stack_depth)
        ]

        for dilation in dilation_layers:
            output = layer(output, conditioning, filters, dilation, mode)
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

        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss,
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
