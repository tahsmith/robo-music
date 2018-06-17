samples_per_second = 44100
channels = 1

n_epochs = 200
batch_size = 2
slice_size = 1225

coding_size = 50

from . import models
import tensorflow as tf

# model = models.DeepConvModel(
#     slice_size,
#     [15,  1,   15,  1,   7,    1,   7, ],
#     [7,   1,   7,   1,   3,    1,   3, ],
#     [40,  40,  80,  80,  160,  160, 320],
#     ["SAME", "SAME", "SAME", "SAME", "SAME", "SAME", "SAME"],
#     [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu],
#     None
# )

model = models.DeepConvModel(
    slice_size,
    [15,],
    [7, ],
    [1, ],
    ["SAME",],
    [tf.nn.elu,],
    None
)