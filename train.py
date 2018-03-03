import datetime
import tensorflow as tf
import numpy as np


def optimiser(model, batches):
    batches = model.prepare(batches)
    optimiser = tf.train.AdamOptimizer()
    cost = model.cost(batches)
    codings = model.encoder(batches)
    reconstructed = model.decoder(codings)
    abs_error = tf.reduce_mean(tf.abs(batches - reconstructed))
    avg = tf.reduce_mean(tf.abs(batches))
    relative_error = abs_error / avg
    op = optimiser.minimize(cost)
    return op, cost, {
        'relative error': relative_error
    }


def train(model):
    test = np.load('./data/test.npy')
    train = np.load('./data/train.npy')

    n_train = train.shape[0]
    n_epochs = 200
    batch_size = 4000
    batches = n_train // batch_size + 1
    x = tf.placeholder(tf.float32, [None, model.slice_size, 1])
    op, cost, misc = optimiser(model, x)

    tag = f'{model.__class__.__name__}' \
          f'-{datetime.datetime.utcnow():%Y_%m_%d_%H_%M}'

    path = './save/' + tag
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    test_cost_summary = tf.summary.scalar('test cost', cost)
    train_cost_summary = tf.summary.scalar('train cost', cost)
    misc_summaries = [tf.summary.scalar(label, var) for label, var in
                      misc.items()]

    file_writer = tf.summary.FileWriter(
        './logs/' + tag,
        tf.get_default_graph()
    )

    log_period = batches * batch_size // 10
    with tf.Session() as session:
        session.run(init)
        for epoch in range(n_epochs):
            print(f'Epoch {epoch + 1}')

            log_counter = 0
            np.random.shuffle(train)
            for i in range(batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_train)
                if start == end:
                    return
                batch = train[start:end, :]
                session.run(op, feed_dict={
                    x: batch,
                    **model.training_feeds()
                })

                log_counter += batch_size
                if log_counter > log_period:
                    log_counter = 0
                    train_cost = session.run(cost, feed_dict={
                        x: batch,
                        **model.testing_feeds()
                    })
                    step = epoch * batches + i
                    string = session.run(train_cost_summary, feed_dict={
                        x: batch,
                        cost: train_cost,
                        **model.testing_feeds()
                    })
                    file_writer.add_summary(string, step)
                    test_cost = session.run(cost, feed_dict={
                        x: test,
                        **model.testing_feeds()
                    })
                    strings = session.run(
                        [test_cost_summary] + misc_summaries,
                        feed_dict={
                            x: test,
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
    import config
    train(config.model)
