

def train(estimator, steps, steps_between_evals, train_input_fn, test_input_fn):
    for i in range(steps ):
        estimator.train(
            input_fn=train_input_fn,
            steps=1000,
        )
        estimator.evaluate(input_fn=test_input_fn, steps=1)