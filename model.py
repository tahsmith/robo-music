import tensorflow as tf

timeslice_size = 1225
samples_per_second = 44100


def Model(inputs):
    conv_layer_config = [
        (5, 2, 2),
    ]

    batches = tf.shape(inputs)[0]
    reshaped_inputs = tf.reshape(
        inputs,
        (batches, 1, -1, 1)
    )
    conv_kernels = []
    input_shapes = [tf.shape(reshaped_inputs)[2]]
    input_depths = [1]
    output_depths = []
    strides = []
    widths = []
    encode_ops = [reshaped_inputs]
    for config in conv_layer_config:
        width, stride, output_depth = config
        output_depths.append(output_depth)
        widths.append(width)
        strides.append(stride)
        conv_kernels.append(
            tf.Variable(tf.random_normal((1, width, input_depths[-1],
                                          output_depth))))
        input_depths.append(output_depth)
        encode_ops.append(tf.nn.conv2d(
            encode_ops[-1],
            conv_kernels[-1],
            strides=[1, 1, stride, 1],
            padding='VALID'
        ))
        input_shapes.append(tf.shape(encode_ops[-1])[2])

    input_shapes = input_shapes[:-1]
    input_depths = input_depths[:-1]
    encoded = tf.reshape(encode_ops[-1], (batches, -1))

    encoded_reshaped = tf.reshape(encoded, (batches, 1, -1, output_depths[-1]))
    output = encoded_reshaped
    for i in reversed(range(len(conv_kernels))):
        output = tf.nn.conv2d_transpose(
            output,
            conv_kernels[i],
            [batches, 1, input_shapes[i], input_depths[i]],
            strides=[1, 1, strides[i], 1],
            padding='VALID'
        )
    decoded = tf.reshape(output, (batches, -1))

    return encoded, decoded
