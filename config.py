
import tensorflow as tf

n_epochs = 200
batch_size = 200
slice_size = 1225
channels = 1

import models

fc_stack = None
fc_stack = models.FcStack(
    960, 50,
    depth=2
)

model = models.DeepConvModel(
    slice_size,
    [15,  1,   15,  1,   7,    1,   7, ],
    [7,   1,   7,   1,   3,    1,   3, ],
    [40,  40,  80,  80,  160,  160, 320],
    ["SAME", "SAME", "SAME", "SAME", "SAME", "SAME", "SAME"],
    [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu],
    fc_stack
)

# model = models.FcStack(
#     slice_size, 50,
#     depth=4
# )