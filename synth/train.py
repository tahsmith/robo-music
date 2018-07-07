import datetime
import tensorflow as tf
import numpy as np
from utils import shuffle


def training_ops(model, x_batch, y_batch):
    batches = model.prepare(x_batch)
    optimiser = tf.train.AdamOptimizer()
    cost = model.cost(batches, y_batch)
    y_pred = model.predict(batches)
    accuracy_op = tf.reduce_mean(
        tf.reduce_sum(tf.abs(y_pred - y_batch), axis=1)
        / tf.reduce_sum(tf.abs(y_batch), axis=1)
    )
    op = optimiser.minimize(cost)
    return op, cost, {
        'accuracy': accuracy_op
    }


def train(batch_size, n_epochs, model_fn, test_data, train_data, model_dir):
    x_test, y_test = test_data
    x_train, y_train = train_data

    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': x_train},
        y_train,
        shuffle=True,
        batch_size=1000,
        num_epochs=None,
    )

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': x_test},
        y_test,
        shuffle=True,
        batch_size=1000,
        num_epochs=None,
    )

    for i in range(100):
        estimator.train(
            input_fn=train_input_fn,
            steps=1000,
        )
        estimator.evaluate(input_fn=test_input_fn, steps=1)


def main():
    from config import config_dict
    config = config_dict['synth']

    all_test = np.load('./cache/synth/x_test.npy')
    x_test = all_test[:, 0:-1]
    y_test = all_test[:, -1]
    all_train = np.load('./cache/synth/x_train.npy')
    x_train = all_train[:, 0:-1]
    y_train = all_train[:, -1]

    model_fn = tf.estimator.LinearRegressor(
        [tf.feature_column.numeric_column(
            'x',
            shape=(
                config_dict['synth']['slice_size'] - 1,
                config_dict['audio']['channels']
            )
        )]
    ).model_fn

    train(
        config['batch_size'],
        config['epoch_count'],
        model_fn,
        (x_test, y_test),
        (x_train, y_train),
        config_dict['data']['logs']
        + f'/synth/{datetime.datetime.utcnow():%Y_%m_%d_%H_%M}'
    )


if __name__ == '__main__':
    main()
