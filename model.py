import tensorflow as tf

timeslice_size = 1225
samples_per_second = 44100


def model():
    layer_1_w = tf.Variable(tf.random_normal((1, 3, 1, 1)))

    def encoder(inputs):
        batches = tf.shape(inputs)[0]
        reshaped_inputs = tf.reshape(
            inputs,
            (batches, 1, timeslice_size, 1)
        )
        layer_1 = tf.nn.conv2d(
            reshaped_inputs,
            layer_1_w,
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        encoded = tf.reshape(layer_1, (batches, -1))
        return encoded

    def decoder(encoded):
        batches = tf.shape(encoded)[0]
        encoded_reshaped = tf.reshape(encoded, (batches, 1, -1, 1))
        conv_transposed = tf.nn.conv2d_transpose(
            encoded_reshaped,
            layer_1_w,
            [batches, 1, timeslice_size, 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        decoded = tf.reshape(conv_transposed, (batches, -1))
        return decoded

    return encoder, decoder
