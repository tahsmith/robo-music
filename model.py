import tensorflow as tf

timeslice_size = 1225
samples_per_second = 44100


def model(inputs):
    conv_layer_config = [
        (5, 2),
        (5, 2),
        (5, 2)
    ]

    conv_kernels = [
        tf.Variable(tf.random_normal((1, width, 1, 1)))
        for width, _ in conv_layer_config
    ]

    batches = tf.shape(inputs)[0]
    reshaped_inputs = tf.reshape(
        inputs,
        (batches, 1, -1, 1)
    )
    output = reshaped_inputs

    output_shapes = []
    for kernel, (_, stride) in zip(conv_kernels, conv_layer_config):
        output_shapes.append(tf.shape(output)[2])
        output = tf.nn.conv2d(
            output,
            kernel,
            strides=[1, 1, stride, 1],
            padding='VALID'
        )

    encoded = tf.reshape(output, (batches, -1))

    batches = tf.shape(encoded)[0]
    encoded_reshaped = tf.reshape(encoded, (batches, 1, -1, 1))
    output = encoded_reshaped
    for kernel, (_, stride), output_shape in reversed(list(zip(conv_kernels,
                                                               conv_layer_config,
                                                               output_shapes))):
        output = tf.nn.conv2d_transpose(
            output,
            kernel,
            [batches, 1, output_shape, 1],
            strides=[1, 1, stride, 1],
            padding='VALID'
        )
    decoded = tf.reshape(output, (batches, -1))

    return encoded, decoded
