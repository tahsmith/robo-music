import tensorflow as tf


def model_fn(features, labels, mode, params):
    with tf.variable_scope('synth'):
        waveform = features['waveform']
        if params['conditioning']:
            conditioning = features['conditioning']
            conditioning = tf.reshape(conditioning, [-1, 128])
            conditioning = tf.layers.dense(conditioning, 128)
            conditioning = tf.reshape(conditioning, [-1, 1, 128])

        else:
            conditioning = None

        input_width = params['slice_size'] - 1
        channels = params['channels']
        residual_filters = params['residual_filters']
        conv_filters = params['conv_filters']
        skip_filters = params['skip_filters']
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
                                  filters=residual_filters)

        dilation_layers = [
            2 ** i
            for _ in range(dilation_stack_count)
            for i in range(dilation_stack_depth)
        ]
        layers = []

        for dilation in dilation_layers:
            output, skip = conv_layer(output, conditioning, residual_filters,
                                      conv_filters, skip_filters, dilation,
                                      mode)
            if dropout:
                output = tf.layers.dropout(
                    output,
                    rate=dropout,
                    training=mode == tf.estimator.ModeKeys.TRAIN
                )
            layers.append(skip)

        output_width = input_width - sum(dilation_layers) - 1
        output = tf.add_n([layer[:, -output_width:, :] for layer in layers])

        with tf.name_scope('fc_stack'):
            flatten = tf.layers.flatten(output)
            dense1 = tf.layers.dense(flatten, quantisation, activation=tf.nn.elu,
                                     name='dense1')
            logits = tf.layers.dense(dense1, quantisation,
                                     name='dense2')

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


def conv_layer(inputs, conditioning_inputs, filters, conv_filters,
               skip_filters, dilation, mode):
    with tf.name_scope('conv_layer'):
        with tf.name_scope('filter'):
            filter_ = conv1d(inputs, conv_filters, dilation)
            if conditioning_inputs is not None:
                filter_ = add_conditioning(filter_, conditioning_inputs,
                                           filters)
            filter_ = tf.tanh(filter_)

        with tf.name_scope('gate'):
            gate = conv1d(inputs, conv_filters, dilation)
            if conditioning_inputs is not None:
                gate = add_conditioning(gate, conditioning_inputs, filters)
            gate = tf.sigmoid(gate)

        layer_outputs = tf.layers.conv1d(filter_ * gate, filters, 1)
        residual = layer_outputs + inputs[:, dilation:, :]
        skip_outputs = tf.layers.conv1d(filter_ * gate, skip_filters, 1)

        return residual, skip_outputs


def add_conditioning(inputs, conditioning, filters):
    conditioning_layer = tf.layers.dense(conditioning, filters)
    outputs = inputs + conditioning_layer
    return outputs


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
        'slice_size': synth_config['slice_size'],
        'channels': audio_config['channels'],
        'dilation_stack_depth': synth_config['dilation_stack_depth'],
        'dilation_stack_count': synth_config['dilation_stack_count'],
        'residual_filters': synth_config['residual_filters'],
        'conv_filters': synth_config['conv_filters'],
        'skip_filters': synth_config['skip_filters'],
        'quantisation': synth_config['quantisation'],
        'regularisation': synth_config['regularisation'],
        'dropout': synth_config['dropout'],
        'conditioning': synth_config['conditioning']
    }
