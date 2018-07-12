import tensorflow as tf


def train_and_test(estimator: tf.estimator.Estimator, steps, steps_between_evals,
                   train_input_fn,
           test_input_fn):
    for i in range(steps // steps_between_evals):
        estimator.train(
            input_fn=train_input_fn,
            steps=steps_between_evals,
        )
        estimator.evaluate(input_fn=test_input_fn)
