import tensorflow as tf


def train_and_test(estimator: tf.estimator.Estimator, train_input_fn,
                   test_input_fn, steps, steps_between_evals, eval_steps):
    eval_results = estimator.evaluate(input_fn=test_input_fn,
                                      steps=eval_steps)
    print(eval_results)

    for i in range(steps // steps_between_evals):
        estimator.train(
            input_fn=train_input_fn,
            steps=steps_between_evals,
        )
        eval_results = estimator.evaluate(input_fn=test_input_fn,
                                        steps=eval_steps)
        print(eval_results)
