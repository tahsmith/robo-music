import synth
from synth.transform import waveform_to_codings
from synth.utils import read_raw_waveform
import glob
import synth.config
from utils import whole_multiple
from . import config
import numpy as np


def count_samples(input_file_list):
    total_size = 0
    for input_file in input_file_list:
        with open(input_file, 'rb') as f:
            size = f.seek(0, 2)
            size //= 2  # to i16
            size = whole_multiple(size, synth.config.slice_size)
            size //= synth.config.slice_size
            size *= synth.config.coding_size * 4  # to f32
            total_size += size
    return total_size


def encode_raw_files(input_file_list):
    total_codings = np.empty((0, synth.config.coding_size), dtype=np.float32)
    for input_file in input_file_list:
        waveform = read_raw_waveform(input_file)
        codings = waveform_to_codings(waveform)
        total_codings = np.concatenate((total_codings, codings), axis=0)
    return total_codings


def main():
    input_file_list = glob.glob('./data/music/*.raw')
    print('{} Files'.format(len(input_file_list)))
    raw = encode_raw_files(input_file_list)
    print('{} MB'.format(raw.shape[0] * raw.shape[1] * 4 // 2 ** 10 / 10 ** 3))
    np.save('./cache/composer/raw.npy', raw, allow_pickle=False)


if __name__ == '__main__':
    main()
