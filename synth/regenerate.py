import os
import sys
import tensorflow as tf
from tensorflow.contrib import ffmpeg
from . import config, transform
from utils import load_sound_file, write_sound_file, normalise_to_int_range
import numpy as np


def regenerate(file_name):
    waveform = load_sound_file(file_name, np.int16,
                               config.samples_per_second,
                               config.channels)
    codings = transform.waveform_to_codings(waveform)
    regenerated = transform.codings_to_waveform(codings)
    regenerated = normalise_to_int_range(regenerated, np.int16)
    *base_name, extn = file_name.split('.')
    base_name = '.'.join(base_name)
    regenerated_file_name = base_name + '_regenerated.' + extn
    write_sound_file(regenerated, regenerated_file_name,
                     config.samples_per_second)


if __name__ == '__main__':
    regenerate(sys.argv[1])
