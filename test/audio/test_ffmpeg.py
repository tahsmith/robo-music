import numpy as np

from audio import ffmpeg


def write_test_file(path):
    sample_rate = 44100
    n = 44100 * 10
    ones = np.ones((n,), dtype=np.int16)
    waveform = np.stack(
        [ones, ones * 2], axis=-1
    )
    ffmpeg.write_sound_file(waveform, path, sample_rate)

    return waveform


def test_write_sound_file(tmpdir):
    write_test_file(tmpdir + '/test.mp3')


def test_load_sound_file(tmpdir):
    file_name = tmpdir + '/test.mp3'
    waveform_expected = write_test_file(file_name)
    waveform_actual = ffmpeg.load_sound_file(file_name, np.int16, 44100, 2)
    assert waveform_expected.shape == waveform_actual.shape

    err = (np.sum(np.abs(waveform_actual - waveform_actual))
           / waveform_expected.shape[0]
           / waveform_expected.shape[1])

    assert err < 1e-9
