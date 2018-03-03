
import tensorflow as tf

n_epochs = 200
batch_size = 2
slice_size = 1225
channels = 1

import models

fc_stack = models.LinearModel(
    24500, 500
)

model = models.DeepConvModel(
    slice_size,
    [251, 1],
    [1, 1],
    [20, 20],
    ["SAME", "SAME"],
    [tf.nn.elu, None],
    fc_stack
)

# model = models.LinearModel(
#     slice_size,
#     slice_size // 4
# )
