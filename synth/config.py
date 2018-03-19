from . import models

n_epochs = 200
batch_size = 10000
slice_size = 1225
channels = 1


model = models.FcStack(
    1226, 1226,
    depth=2
)
