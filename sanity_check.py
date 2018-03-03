import tensorflow as tf
import numpy as np
import prepare
from config import model, channels, slice_size

raw = tf.Variable(np.random.randn(2, slice_size, channels),
                  dtype=tf.float32)

encoded = model.encoder(model.prepare(raw))
decoded = model.decoder(encoded)
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    raw_values = np.random.randn(2, prepare.slice_size, channels)
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
    assert (raw_values.shape == decoded_values.shape)
