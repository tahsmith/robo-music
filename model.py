import tensorflow as tf

timeslice_size = 1224
samples_per_second = 44100


def Model(inputs, width, depth, batches):
    conv_layer_config = [
        (1, 1, 2, tf.nn.relu),
        (5, 2, 12, tf.nn.relu),
    ]

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
    activations = []
    encode_ops = [reshaped_inputs]
    for config in conv_layer_config:
        width, stride, output_depth, activation = config
        output_depths.append(output_depth)
        widths.append(width)
        strides.append(stride)
        activations.append(activation)
        conv_kernels.append(
            tf.Variable(tf.random_normal((1, width, input_depths[-1],
                                          output_depth))))
        conv_biases.append(
            tf.Variable(tf.zeros((output_depth,)))
        )
        input_depths.append(output_depth)
        op = tf.nn.conv2d(
            encode_ops[-1],
            conv_kernels[-1],
            strides=[1, 1, stride, 1],
            padding='SAME'
        ) + conv_biases[-1]
        if activation:
            op = activation(op)
        encode_ops.append(op)
        input_shapes.append(tf.shape(encode_ops[-1])[2])

    input_shapes = input_shapes[:-1]
    input_depths = input_depths[:-1]

    flatten = tf.reshape(encode_ops[-1], (batches, -1))
    # Add fully connected layer

    fc_layer_1_w = tf.Variable(tf.random_normal([int(flatten.shape[1]), 500]))
    fc_layer_1_b = tf.Variable(tf.zeros(500))
    fc_layer_1 = tf.matmul(flatten, fc_layer_1_w) + fc_layer_1_b
    encoded = fc_layer_1

    fc_layer_1_rev = tf.matmul(encoded - fc_layer_1_b, tf.transpose(fc_layer_1_w))

    encoded_reshaped = tf.reshape(fc_layer_1_rev, (batches, 1, -1,
                                                   output_depths[-1]))
    output = encoded_reshaped
    for i in reversed(range(len(conv_kernels))):
        if activations[i]:
            output = activations[i](output)
        output = tf.nn.conv2d_transpose(
            output - conv_biases[i],
            conv_kernels[i],
            [batches, 1, input_shapes[i], input_depths[i]],
            strides=[1, 1, strides[i], 1],
            padding='SAME'
        )
    decoded = tf.reshape(output, (batches, -1, depth))

    return encoded, decoded
