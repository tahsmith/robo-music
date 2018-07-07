import sys
import tensorflow as tf
import numpy as np

from .model import model


def main(argv):
    estimator = tf.estimator.Estimator(model, 'logs/classifier')

    x_train = np.load('./cache/classifier/x_train.npy')
    y_train = np.load('./cache/classifier/y_train.npy')
    x_test = np.load('./cache/classifier/x_test.npy')
    y_test = np.load('./cache/classifier/y_test.npy')

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
        shuffle=True
    )

    for i in range(100):
        estimator.train(
            input_fn=train_input_fn,
            steps=1000,
        )
        estimator.evaluate(input_fn=test_input_fn, steps=1)


if __name__ == '__main__':
    main(sys.argv)
