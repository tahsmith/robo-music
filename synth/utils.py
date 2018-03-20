import numpy as np

from . import config


def read_raw_waveform(raw_waveform_file_name, dtype='<i2'):
    waveform = np.fromfile(raw_waveform_file_name, dtype=dtype)
    waveform = np.array(waveform, dtype=np.float32)
    waveform = np.reshape(waveform, (-1, config.channels))
    return waveform