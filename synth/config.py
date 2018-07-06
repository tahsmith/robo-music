samples_per_second = 11025
channels = 1

n_epochs = 400
batch_size = 500
slice_size = 2 ** 10

from . import models
import tensorflow as tf

model = models.DeepConvModel(
    slice_size,
    [2, 2, 2, 2, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2, 2, 2, 2],
    [32, 32, 32, 32, 64, 64, 64, 64, 64],
    None,
    None
)
