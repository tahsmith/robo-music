import glob
from itertools import product
import subprocess

import numpy as np
from utils import conv_size
from config import slice_size, channels, batch_size


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


def main():
    files = glob.glob('./data/*.raw')
    np.random.shuffle(files)
    all_data = np.zeros((0, 1), np.float32)
    for file in files:
        waveform = np.fromfile(file, dtype='<i2')
        waveform = np.array(waveform, dtype=np.float32)
        waveform = np.reshape(waveform, (-1, channels))
        waveform = (waveform - np.mean(waveform)) / np.std(waveform)
        all_data = np.concatenate((all_data, waveform), axis=0)

    # all_data = all_data[:125650, :]
    # Cut down to percent chunks
    n_samples = all_data.shape[0]
    chunk_size = n_samples // 100
    all_data = all_data[:n_samples - n_samples % 100, :]
    percent_chunks = all_data.reshape((-1, chunk_size, channels))
    train_percent = 90
    stride = slice_size // 4
    slice_per_chunk = conv_size(chunk_size, slice_size, stride, 'VALID')

    print('Total samples ', n_samples)
    print('Chunk size    ', chunk_size)
    print('Chunked size  ', chunk_size * 100)
    print('Chunking loss ', (n_samples - 100 * chunk_size) * 100 / n_samples,
          '%')
    train_count = train_percent * slice_per_chunk
    print('Train count   ', train_count)
    print('Train size    ',
          train_count * channels * slice_size * 4 / 2 ** 10 / 1e3, 'MB')

    test_count = (100 - train_percent) * slice_per_chunk
    print('Test count    ', test_count)
    print('Test size     ',
          test_count * channels * slice_size * 4 / 2 ** 10 / 1e3, 'MB')

    np.random.shuffle(percent_chunks)
    train_chunks = percent_chunks[:train_percent, :, :]
    test_chuncks = percent_chunks[train_percent:, :, :]

    train = batch(generate_slice_set(train_chunks, stride), )
    print(train.shape)
    save_array(train, './data/train.npy')
    test = batch(generate_slice_set(test_chuncks, stride))
    print(test.shape)
    save_array(test, './data/test.npy')


if __name__ == '__main__':
    main()
