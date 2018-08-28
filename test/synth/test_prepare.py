import numpy as np

from synth.model import ModelParams
from synth.prepare import compute_features


def test_compute_features(params):
    sample_rate = 44100
    waveform = np.random.normal(0, 1, (sample_rate * 2, 1))
    features = compute_features(waveform, params)

    assert waveform.shape[0] == features.shape[0]
    assert np.all(features < 1.0)
