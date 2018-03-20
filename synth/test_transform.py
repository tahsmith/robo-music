from . import transform
from . import config
import numpy as np


def test_wave_form_to_coding():
    waveform = np.random.randint(
        - 2 ** 15, 2 ** 15,
        size=(
            np.random.randint(config.slice_size, 2 * config.slice_size),
            config.channels),
    )
    codings = transform.waveform_to_codings(waveform)

    assert codings.shape[1] == config.coding_size
