import numpy as np

from synth.prepare import compute_features


def test_compute_features():
    sample_rate = 44100
    feature_window = 2048
    n_mels = 128
    waveform = np.random.normal(0, 1, (sample_rate * 2, 1))
    features = compute_features(waveform, sample_rate, feature_window, n_mels)

    assert waveform.shape[0] == features.shape[0]
    assert np.all(features < 1.0)
