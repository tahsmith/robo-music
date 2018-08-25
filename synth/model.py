import tensorflow as tf
from attr import attrs, attrib


@attrs
class ModelParams:
    slice_size = attrib()
    channels = attrib()
    dilation_stack_depth = attrib()
    dilation_stack_count = attrib()
    residual_filters = attrib()
    conv_filters = attrib()
    skip_filters = attrib()
    quantisation = attrib()
    regularisation = attrib()
    dropout = attrib()
    conditioning = attrib()
    sample_rate = attrib()
    feature_window = attrib()
    n_mels = attrib()

    @property
    def receptive_field(self):
        return model_width(self.dilation_stack_depth, self.dilation_stack_count)


def model_fn(features, mode, params):
    with tf.variable_scope('synth'):
        conditioning, input_waveform, waveform = init_features(features, mode,
                                                               params)

        assert params.channels == 1
        one_hot = tf.one_hot(
            input_waveform[:, :, 0],
            params.quantisation
        )

        input_width = tf.shape(input_waveform)[1]
        one_hot = tf.reshape(one_hot, [-1, input_width, params.quantisation])

        with tf.variable_scope('input_reshape'):
            output = tf.layers.conv1d(one_hot, kernel_size=2, strides=1,
                                      filters=params.residual_filters)
            if conditioning is not None:
                conditioning = conditioning[:, 1:, :]


        dilation_layers = [
            2 ** i
            for _ in range(params.dilation_stack_count)
            for i in range(params.dilation_stack_depth)
        ]
        layers = []

        for n, dilation in enumerate(dilation_layers):
            with tf.variable_scope(f'layer_{n}'):
                output, skip, conditioning = conv_layer(output, conditioning,
                                                        dilation, params)
                if params.dropout:
                    output = tf.layers.dropout(
                        output,
                        rate=params.dropout,
                        training=mode == tf.estimator.ModeKeys.TRAIN
                    )
                layers.append(skip)

        output_width = input_width - sum(dilation_layers) - 1
        output = sum([layer[:, -output_width:, :] for layer in layers])

        with tf.variable_scope('fc_stack'):
            output = tf.nn.elu(output)
            output = tf.layers.conv1d(output, params.skip_filters, 1,
                                      activation=tf.nn.elu)
            logits = tf.layers.conv1d(output, params.quantisation, 1,
                                      activation=None)

        predictions = tf.argmax(logits, axis=2, output_type=tf.int32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode, predictions,
                export_outputs={
                    'predict_output': tf.estimator.export.PredictOutput(
                        {'predictions': predictions,
                         'probabilities': tf.nn.softmax(logits)})})

        n_predictions = tf.shape(logits)[1]
        target_waveform = waveform
        labels = target_waveform[:, -n_predictions:]
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)

        loss = add_regularisation(loss, params)

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(labels, predictions),
                'mean_absolute_error':
                    tf.metrics.mean_absolute_error(labels, predictions)
            }
            return tf.estimator.EstimatorSpec(
                mode,
                predictions,
                loss,
                eval_metric_ops=eval_metric_ops
            )

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer()
            training_op = optimizer.minimize(loss,
                                             tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(
                mode,
                predictions,
                loss,
                training_op
            )


def add_regularisation(loss, params):
    if params.regularisation:
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           'synth')

        reg_loss = sum([tf.nn.l2_loss(x) for x in trainable_vars])
        loss += params.regularisation * reg_loss
    return loss


def init_features(features, mode, params):
    waveform = features['waveform']
    if mode == tf.estimator.ModeKeys.PREDICT:
        input_waveform = waveform
    else:
        input_waveform = waveform[:, :-1, :]

    if params.conditioning:
        if mode == tf.estimator.ModeKeys.PREDICT:
            out_size = params.slice_size
            conditioning = features['conditioning']
        else:
            out_size = params.slice_size - 1
            conditioning = features['conditioning'][:, :-1, :]
        conditioning = tf.reshape(conditioning, (params.batch_count,
                                                 out_size, params.n_mels))
    else:
        conditioning = None

    return conditioning, input_waveform, waveform


def conv_layer(inputs, conditioning_inputs, dilation, params):
    with tf.variable_scope('conv_layer'):
        with tf.variable_scope('filter'):
            filter_ = conv1d_with_conditioning(inputs, conditioning_inputs,
                                               dilation, params)
            filter_ = tf.tanh(filter_)

        with tf.variable_scope('gate'):
            gate = conv1d_with_conditioning(inputs, conditioning_inputs,
                                            dilation, params)
            gate = tf.sigmoid(gate)

        dilation_output = filter_ * gate

        # TODO: is this necessary?
        layer_outputs = tf.layers.conv1d(dilation_output,
                                         params.residual_filters, 1)

        with tf.variable_scope('residual'):
            residual = layer_outputs + inputs[:, dilation:, :]

        with tf.variable_scope('skip'):
            skip_outputs = tf.layers.conv1d(dilation_output,
                                            params.skip_filters, 1)

        if conditioning_inputs is not None:
            reduced_conditioning = conditioning_inputs[:, dilation:, :]
        else:
            reduced_conditioning = None

        return residual, skip_outputs, reduced_conditioning


def conv1d_with_conditioning(inputs, conditioning_inputs, dilation, params):
    filter_ = conv1d(inputs, params.conv_filters, dilation)
    if conditioning_inputs is not None:
        filter_ = add_conditioning(filter_, conditioning_inputs,
                                   params.conv_filters, dilation)
    return filter_


def add_conditioning(inputs, conditioning, filters, dilation):
    with tf.variable_scope('conditioning_conv'):
        conditioning_layer = conv1d(conditioning, filters, dilation)
        outputs = inputs + conditioning_layer
    return outputs


def conv1d(inputs, filters, dilation):
    return tf.layers.conv1d(inputs,
                            filters=filters,
                            kernel_size=2,
                            dilation_rate=dilation,
                            padding='valid')


def dilation_stack_width(depth, count):
    return sum(
        2 ** i
        for _ in range(count)
        for i in range(depth)
    )


def model_width(depth, count):
    return dilation_stack_width(depth, count) + 2


def params_from_config():
    from config import config_dict
    audio_config = config_dict['audio']
    synth_config = config_dict['synth']
    return ModelParams(
        slice_size=synth_config['slice_size'],
        channels=audio_config['channels'],
        dilation_stack_depth=synth_config[
            'dilation_stack_depth'],
        dilation_stack_count=synth_config[
            'dilation_stack_count'],
        residual_filters=synth_config['residual_filters'],
        conv_filters=synth_config['conv_filters'],
        skip_filters=synth_config['skip_filters'],
        quantisation=synth_config['quantisation'],
        regularisation=synth_config['regularisation'],
        dropout=synth_config['dropout'],
        conditioning=synth_config['conditioning'],
        sample_rate=audio_config['sample_rate'],
        feature_window=synth_config['feature_window'],
        n_mels=128
    )
