import tensorflow as tf
import numpy as np
from model import model, timeslice_size

raw = tf.Variable(np.random.randn(2, timeslice_size), dtype=tf.float32)

encoded, decoded = model(raw)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    raw_values = np.random.randn(2, timeslice_size)
    print(raw_values.shape)
    encoded_values = session.run(
        encoded,
        feed_dict={
            raw: raw_values
        }
    )
    print(encoded_values.shape)
    decoded_values = session.run(
        decoded,
        feed_dict={
            encoded: np.random.randn(*encoded_values.shape)
        }
    )
    print(decoded_values.shape)
    assert(raw_values.shape == decoded_values.shape)
