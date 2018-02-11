import tensorflow as tf

timeslice_size = 1224
samples_per_second = 44100


def Model(inputs, width, depth):
    conv_layer_config = [
        (1, 1, 2, tf.nn.relu),
        (5, 1, 1, None),
    ]

    input_shape = tf.shape(inputs)
    batches = input_shape[0]

    reshaped_inputs = tf.reshape(
        inputs,
        (batches, 1, width, depth)
    )
    conv_kernels = []
    conv_biases = []
    input_shapes = [tf.shape(reshaped_inputs)[2]]
    input_depths = [depth]
    output_depths = []
    strides = []
    widths = []
    encode_ops = [reshaped_inputs]
    for config in conv_layer_config:
        width, stride, output_depth, activation = config
        output_depths.append(output_depth)
        widths.append(width)
        strides.append(stride)
        conv_kernels.append(
            tf.Variable(tf.random_normal((1, width, input_depths[-1],
                                          output_depth))))
        conv_biases.append(
            tf.Variable(tf.zeros((output_depth,)))
        )
        input_depths.append(output_depth)
        op = tf.nn.relu(tf.nn.conv2d(
            encode_ops[-1],
            conv_kernels[-1],
            strides=[1, 1, stride, 1],
            padding='VALID'
        ) + conv_biases[-1])
        if activation:
            op = activation(op)
        encode_ops.append(op)
        input_shapes.append(tf.shape(encode_ops[-1])[2])

    input_shapes = input_shapes[:-1]
    input_depths = input_depths[:-1]

    flatten = tf.reshape(encode_ops[-1], (batches, -1))
    # Add fully connected layer
    encoded = flatten

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
    decoded = tf.reshape(output, (batches, -1, depth))

    return encoded, decoded
