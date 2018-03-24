samples_per_second = 44100
channels = 1

n_epochs = 200
batch_size = 2000
slice_size = 1224

feature_size = 2 * (slice_size // 2 + 1)
coding_size = 50

from . import models
import tensorflow as tf

model = models.FcStack(
    feature_size, coding_size,
    depth=5,
    reuse=False
)
