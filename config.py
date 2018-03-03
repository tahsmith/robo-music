import models
import tensorflow as tf

batch_size = 2
slice_size = 1225
channels = 1

# model = models.DeepConvModel(
#         slice_size,
#         [125, 1, 25, 1,  15, 1,  5],
#         [1,   1, 13, 1,  7,  1,  3],
#         [10,  1, 10, 10, 20, 20, 30],
#         ["SAME", "SAME", "SAME", "SAME", "SAME", "SAME", "SAME"],
#         [tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu, tf.nn.elu,
#          tf.nn.elu]
#     )

model = models.LinearModel(
    slice_size,
    slice_size // 4
)
