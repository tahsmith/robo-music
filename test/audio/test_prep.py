import numpy as np

from audio.prep import (
    waveform_to_slices
)


def test_waveform_to_samples():
    waveform = np.arange(0, 6).reshape((-1, 1))

    slices = waveform_to_slices(waveform, 4, 2)
    assert (slices == [[[0], [1], [2], [3]], [[2], [3], [4], [5]]]).all()


def test_waveform_to_samples_odd():
    waveform = np.arange(0, 5).reshape((-1, 1))

    slices = waveform_to_slices(waveform, 4, 2)
    assert (slices == [[[0], [1], [2], [3]]]).all()
