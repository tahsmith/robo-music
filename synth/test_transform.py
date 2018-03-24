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


def test_coding_to_waveform():
    codings = np.random.randn(
        np.random.randint(config.slice_size // config.coding_size,
                          2 * config.slice_size // config.coding_size),
        config.coding_size,
    )
    waveform = transform.codings_to_waveform(np.float32(codings))


def test_round_trip():
    waveform_expected = np.random.randint(
        - 2 ** 15, 2 ** 15,
        size=(config.slice_size*2,
            config.channels),
    )
    waveform_actual = transform.codings_to_waveform(transform.waveform_to_codings(
        waveform_expected))

    assert waveform_expected.shape == waveform_actual.shape