import tensorflow as tf
import numpy as np
from model import model, timeslice_size

channels = 1
raw = tf.Variable(np.random.randn(2, timeslice_size, channels), dtype=tf.float32)

encoded, decoded = model(raw, timeslice_size, channels, 2)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    raw_values = np.random.randn(2, timeslice_size, channels)
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
