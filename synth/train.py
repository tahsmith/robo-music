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


def train(batch_size, n_epochs, model, test_data, train_data):
    x_test, y_test = test_data
    x_train, y_train = train_data

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    print(f'Training samples: {n_train}')
    print(f'Test samples: {n_test}')
    batches = n_train // batch_size + 1
    x = tf.placeholder(tf.float32, [None, model.n_features])
    y = x
    op, cost, misc = training_ops(model, x, y)
    misc_summaries = [tf.summary.scalar(label, var) for label, var in
                      misc.items()]

    tag = f'synth' \
          f'-{datetime.datetime.utcnow():%Y_%m_%d_%H_%M}'

    path = './save/' + tag
    saver = tf.train.Saver()

    test_cost_summary = tf.summary.scalar('test cost', cost)
    train_cost_summary = tf.summary.scalar('train cost', cost)

    file_writer = tf.summary.FileWriter(
        './logs/' + tag,
        tf.get_default_graph()
    )

    init = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    log_period = batches * batch_size // 10
    with tf.Session() as session:
        session.run(init)
        for epoch in range(n_epochs):
            print(f'Epoch {epoch + 1}')

            log_counter = 0
            x_train = shuffle(x_train)
            for i in range(batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_train)
                if start == end:
                    return
                x_batch = x_train[start:end]
                y_batch = x_train[start:end]
                session.run(op, feed_dict={
                    x: x_batch,
                    y: y_batch,
                    **model.training_feeds()
                })

                log_counter += batch_size
                if log_counter > log_period:
                    log_counter = 0
                    train_cost = session.run(cost, feed_dict={
                        x: x_batch,
                        y: y_batch,
                        **model.testing_feeds()
                    })
                    step = epoch * batches + i
                    string = session.run(train_cost_summary, feed_dict={
                        x: x_batch,
                        y: y_batch,
                        cost: train_cost,
                        **model.testing_feeds()
                    })
                    file_writer.add_summary(string, step)
                    test_cost = session.run(cost, feed_dict={
                        x: x_test,
                        y: y_test,
                        **model.testing_feeds()
                    })
                    strings = session.run(
                        [test_cost_summary] + misc_summaries,
                        feed_dict={
                            x: x_test,
                            y: y_test,
                            cost: test_cost,
                            **model.testing_feeds()
                        }
                    )
                    for string in strings:
                        file_writer.add_summary(string, step)
                    info = f'{i + 1:>5d} / {batches:<5d}' \
                           f': {train_cost:>2.2g} {test_cost:>2.2g}'
                    print(info)
                    saver.save(session, path)


if __name__ == '__main__':
    from . import config

    x_test = np.load('./cache/synth/x_test.npy')
    # y_test = np.load('./cache/synth/y_test.npy')
    y_test = x_test
    x_train = np.load('./cache/synth/x_train.npy')
    # y_train = np.load('./cache/synth/y_train.npy')
    y_train = x_train
    train(config.batch_size, config.n_epochs, config.model,
          (x_test, y_test),
          (x_train, y_train))
