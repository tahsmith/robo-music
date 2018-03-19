from composer import rnn
import numpy as np


def test_features():
    x = np.random.randn(rnn.slice_size * 10)
    x = rnn.normalise(x)
    features = rnn.waveform_to_features(x)
    x_actual = rnn.features_to_waveform(features)
    assert np.allclose(x, x_actual)
