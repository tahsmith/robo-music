from contexttimer import Timer
import synth.config
import tensorflow as tf
from . import model
from . import config
import numpy as np


X = model.codings
y = tf.placeholder(tf.float32,
                   [None, config.n_steps, config.n_inputs], name='y')

model = model.model

loss = tf.reduce_mean(tf.square(model.outputs - y))

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

n_epochs = 2000
n_iterations = 150
batch_size = 1000

t_min, t_max = 0, 30
resolution = 1 / synth.config.samples_per_second


def next_batch(X_all, batch_size, n_steps):
    start_indices = np.random.randint(0, X_all.shape[0] - n_steps,
                                      size=(batch_size, 1))
    indices = start_indices + np.arange(0, n_steps + 1)
    return (
        X_all[indices[:, :-1], :].reshape(batch_size, n_steps, config.n_inputs),
        X_all[indices[:, 1:], :].reshape(batch_size, n_steps, config.n_inputs)
    )


class EarlyTermination(Exception):
    pass


def main():
    X_all = np.load('./cache/composer/raw.npy', allow_pickle=False)
    print(X_all.dtype, X_all.shape)

    saver = tf.train.Saver()

    with Timer() as t:
        with tf.Session() as sess:
            init.run()
            try:
                for epoc in range(n_epochs):
                    for iteration in range(n_iterations):
                        X_batch, y_batch = next_batch(X_all, batch_size,
                                                      config.n_steps)
                        sess.run(training_op,
                                 feed_dict={X: X_batch, y: y_batch})
                        if iteration % 10 == 0:
                            loss_eval = loss.eval(
                                feed_dict={X: X_batch, y: y_batch})
                            print(iteration, "t = ", t.elapsed, "Loss: ",
                                  loss_eval)
                            saver.save(sess, './save/composer')
                            if loss_eval < 0.001:
                                raise EarlyTermination
                            if t.elapsed > 2500:
                                raise EarlyTermination
            except EarlyTermination:
                pass


if __name__ == '__main__':
    main()
