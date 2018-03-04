import glob
import os
from itertools import product

import numpy as np
from os import walk

from utils import conv_size
from config import slice_size, channels, batch_size

samples_per_second = 44100


def save_array(x, filename):
    np.save(filename, x, allow_pickle=False)


def make_batch(all_data, start, size, timeslice_size):
    indices = np.arange(start, size).reshape((size, 1))
    indices = indices + np.arange(0, timeslice_size)
    return all_data[indices.astype(np.int32)]


def generate_slice_set(chunks, stride):
    chunk_size = chunks.shape[1]
    chunk_count = chunks.shape[0]
    slice_per_chunk = conv_size(chunk_size, slice_size, stride, 'VALID')

    chunk_indices = list(range(chunk_count))
    slice_indices = list(range(slice_per_chunk))
    indices = product(chunk_indices, slice_indices)

    for i_chunk, i_slice in indices:
        begin = i_slice * stride
        end = begin + slice_size
        if end <= chunk_size:
            yield chunks[i_chunk, begin:end, :]


def batch(slices):
    slices = list(slices)
    batch = np.empty((len(slices), slice_size, channels))
    for i, slice in enumerate(slices):
        batch[i, :, :] = slice

    return batch


def list_categories():
    category_names = []
    for root, dirs, files in walk('./data'):
        for dir in dirs:
            category_names.append(dir)
        if root != './data':
            break

    return category_names


def list_input_files():
    return {
        k: glob.glob(os.path.join('./data', k, '*.raw'))
        for k in list_categories()
    }


def concat_raw_from_files(file_list):
    all_data = np.zeros((0, 1), np.float32)
    for file in file_list:
        waveform = np.fromfile(file, dtype='<i2')
        waveform = np.array(waveform, dtype=np.float32)
        waveform = np.reshape(waveform, (-1, channels))
        waveform = (waveform - np.mean(waveform)) / np.std(waveform)
        all_data = np.concatenate((all_data, waveform), axis=0)
    return all_data


def create_samples(waveform, cat):
    waveform = waveform[:waveform.shape[0] - waveform.shape[0] % slice_size, :]
    waveform = waveform.reshape((-1, slice_size, channels))
    return waveform, np.ones((waveform.shape[0],), dtype=np.uint8) * cat


def main():
    i = 0
    all_x = np.empty((0, slice_size, channels), dtype=np.float32)
    all_y = np.empty((0,), dtype=np.uint8)
    for k, v in list_input_files().items():
        all_data_for_cat = concat_raw_from_files(v)
        seconds = all_data_for_cat.shape[0] / samples_per_second

        x, y = create_samples(all_data_for_cat, i)
        assert(x.shape[0] == y.shape[0])
        print(f'{i} {k}: {seconds:0.2f}s {x.shape[0]} samples')
        all_x = np.concatenate((all_x, x), axis=0)
        all_y = np.concatenate((all_y, y), axis=0)
        i += 1

    n_samples = all_x.shape[0]
    indices = np.arange(0, n_samples)
    np.random.shuffle(indices)
    all_x = all_x[indices, :, :]
    all_y = all_y[indices]

    train_percent = 90
    i_train = (n_samples * train_percent) // 100
    x_train = all_x[0:i_train, :, :]
    y_train = all_y[0:i_train]

    x_test = all_x[i_train:, :, :]
    y_test = all_y[i_train:]

    save_array(x_test, './data/x_test.npy')
    save_array(y_test, './data/y_test.npy')
    save_array(x_train, './data/x_train.npy')
    save_array(y_train, './data/y_train.npy')


if __name__ == '__main__':
    main()
