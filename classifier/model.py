import tensorflow as tf

UNITS = 100
CATEGORIES = 10


def model(features, labels, mode):
    x = features['x']
    output = tf.layers.dense(x, UNITS, tf.nn.elu, name='hidden_layer')
    logits = tf.layers.dense(x, CATEGORIES, name='output_layer')
    softmax = tf.nn.softmax(logits)
    prediction = tf.argmax(logits, axis=1)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=output)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        training_op = tf.train.AdamOptimizer().minimize(
            loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=training_op)
