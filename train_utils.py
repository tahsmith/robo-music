import tensorflow as tf


def train_and_test(estimator: tf.estimator.Estimator, train_input_fn,
                   test_input_fn, steps, steps_between_evals, eval_steps):
    for i in range(steps // steps_between_evals):
        estimator.train(
            input_fn=train_input_fn,
            steps=steps_between_evals,
        )
        estimator.evaluate(input_fn=test_input_fn, steps=eval_steps)
