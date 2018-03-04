
import tensorflow as tf

n_epochs = 200
batch_size = 200
slice_size = 1225
channels = 1

import models


model = models.FcStack(
    1226, 4,
    depth=8
)