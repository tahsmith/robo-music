import numpy as np
import tensorflow as tf
from pytest import approx

from synth.train import (
    augment_sample
)


def test_augment_none(sess):
    waveform = np.arange(0, 1000, dtype=np.float32).reshape(100, 10, 1)
    augmented = augment_sample(tf.constant(waveform), 0.0, 1.0)
    augmented_actual = sess.run(augmented)

    assert waveform.shape == augmented_actual.shape
    assert np.all(waveform == augmented_actual)


def test_augment_scale(sess):
    waveform = np.ones((100, 10, 1), dtype=np.float32)
    augmented = augment_sample(tf.constant(waveform), 0.0, 2.0)
    augmented_actual = sess.run(augmented)

    assert waveform.shape == augmented_actual.shape
    assert np.allclose(np.diff(augmented_actual, axis=1), 0.0)


def test_augment_mean(sess):
    waveform = np.zeros((2, 1000000, 1), dtype=np.float32)
    augmented = augment_sample(tf.constant(waveform), 1.0, 0.0)
    augmented_actual = sess.run(augmented)

    assert waveform.shape == augmented_actual.shape
    assert np.allclose(np.mean(augmented_actual, axis=1), 0.0, atol=1e-2)
    assert np.allclose(np.std(augmented_actual, axis=1), 1.0, atol=1e-2)
