import tensorflow as tf

UNITS = 100
CATEGORIES = 10
LAYERS = 5


def stack(inputs, mode):
    outputs = inputs
    outputs = tf.layers.dense(outputs, UNITS, tf.nn.elu)
    outputs = tf.layers.dense(outputs, UNITS, tf.nn.elu)
    outputs = inputs + outputs

    dropout_rate = 0.5 if mode == tf.estimator.ModeKeys.TRAIN else 0.0
    outputs = tf.layers.dropout(outputs, dropout_rate)

    return outputs


def model(features, labels, mode):
    x = features['x']
    output = x
    dropout_rate = 0.5 if mode == tf.estimator.ModeKeys.TRAIN else 0.0
    output = tf.layers.dense(output, UNITS)
    for i in range(LAYERS):
        output = stack(output, mode)

    logits = tf.layers.dense(output, CATEGORIES, name='output_layer')
    softmax = tf.nn.softmax(logits)
    predictions = tf.argmax(logits, axis=1)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    if mode == tf.estimator.ModeKeys.TRAIN:
        training_op = tf.train.AdamOptimizer().minimize(
            loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=training_op)
