n_epochs = 200
batch_size = 10000
slice_size = 1225
channels = 1

import models


model = models.FcStack(
    613, 9,
    depth=8
)