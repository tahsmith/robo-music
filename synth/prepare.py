import glob
import os
from itertools import product

import librosa
import numpy as np
from os import walk
import tensorflow as tf

from utils import conv_size, normalise_to_int_range


def save_array(x, filename):
    np.save(filename, x, allow_pickle=False)


def make_batch(all_data, start, size, timeslice_size):
    indices = np.arange(start, size).reshape((size, 1))
    indices = indices + np.arange(0, timeslice_size)
    return all_data[indices.astype(np.int32)]


def generate_slice_set(chunks, stride, slice_size):
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


def list_categories():
    category_names = []
    root = './data/samples'
    for root, dirs, files in walk(root):
        for dir_ in dirs:
            if dir_[0] != '_':
                category_names.append(dir_)
        if root != root:
            break

    return category_names


def list_input_files():
    return glob.glob('cache/samples/**/*.npy', recursive=True)


def quantise(x, quantisation):
    if quantisation != 256:
        raise NotImplementedError('quantisation != 256')
    return normalise_to_int_range(x, np.uint8).astype(np.int32)


def concat_raw_from_files(file_list, channels):
    all_data = np.zeros((0, 1), np.float32)
    for file in file_list:
        waveform = np.load(file)
        waveform = np.array(waveform, dtype=np.float32)
        waveform = np.reshape(waveform, (-1, channels))
        waveform = (waveform - np.mean(waveform)) / np.std(waveform)
        all_data = np.concatenate((all_data, waveform), axis=0)
    return all_data


def compute_features(waveform, sample_rate, slice_size, stride, n_mels):
    if waveform.shape[1] == 1:
        waveform = waveform.reshape((-1))

    spec = np.abs(librosa.stft(waveform, n_fft=slice_size, hop_length=stride,
                  center=False)) ** 2

    spec = librosa.feature.melspectrogram(
        waveform,
        sample_rate,
        S=spec,
        n_mels=n_mels
    )

    spec = spec.transpose((1, 0)).astype(np.float32)

    return spec


def create_samples(waveform, slice_size, stride):
    channels = waveform.shape[1]
    n_samples = (waveform.shape[0] - slice_size) // stride + 1
    # guard against waveform being too small to make a sample
    n_samples = max(n_samples, 0)
    samples = np.empty((n_samples, slice_size, channels), waveform.dtype)
    for i in range(n_samples):
        begin = i * stride
        end = begin + slice_size
        assert (end <= waveform.shape[0])
        slice_ = waveform[begin:end]
        samples[i, :, :] = slice_
    assert samples.dtype == np.int32
    return samples


def clip_to_slice_size(slice_size, waveform):
    waveform = waveform[
               :waveform.shape[0] - waveform.shape[0] % slice_size, :]
    return waveform


def main():
    from config import config_dict
    slice_size = config_dict['synth']['slice_size']
    quantisation = config_dict['synth']['quantisation']
    channels = config_dict['audio']['channels']
    samples_per_second = config_dict['audio']['sample_rate']
    n_mels = config_dict['classifier']['n_mels']

    i = 0
    all_x = np.empty((0, slice_size, channels), dtype=np.int32)
    all_y = np.empty((0, n_mels), dtype=np.float32)

    input_files = list_input_files()
    # for k, v in input_files.items():
    #     print(f'{k}: {len(v)}')

    # number of data points generated per slice.
    stride = slice_size // 8

    for v in input_files:
        all_data_for_cat = concat_raw_from_files([v], channels)
        seconds = all_data_for_cat.shape[0] / samples_per_second

        cat_x = np.empty((0, slice_size, channels), dtype=np.int32)
        cat_y = np.empty((0, n_mels), dtype=np.float32)
        preprocessing_batch_size = 400000
        for j in range(i, all_data_for_cat.shape[0],
                       preprocessing_batch_size):
            begin = j
            end = min(j + preprocessing_batch_size,
                      all_data_for_cat.shape[0])
            waveform = all_data_for_cat[begin:end]
            waveform = clip_to_slice_size(slice_size, waveform)
            if waveform.shape[0] == 0:
                continue
            y = compute_features(waveform, samples_per_second, slice_size,
                                 stride, n_mels)
            waveform = quantise(waveform, quantisation)
            samples = create_samples(waveform, slice_size, stride)

            assert samples.shape[0] == y.shape[0]

            cat_x = np.concatenate((cat_x, samples), axis=0)
            cat_y = np.concatenate((cat_y, y), axis=0)
            assert cat_x.dtype == np.int32

        all_x = np.concatenate((all_x, cat_x), axis=0)
        all_y = np.concatenate((all_y, cat_y), axis=0)
        print(f'{i} {v}: {seconds:0.2f}s {cat_x.shape[0]} samples')
        i += 1

        assert all_x.dtype == np.int32

    n_samples = all_x.shape[0]
    indices = np.arange(0, n_samples)
    np.random.shuffle(indices)
    all_x = all_x[indices]
    all_y = all_y[indices]

    train_percent = 90
    i_train = (n_samples * train_percent) // 100
    x_train = all_x[0:i_train]
    y_train = all_y[0:i_train]

    x_test = all_x[i_train:]
    y_test = all_y[i_train:]

    print("Summary:")
    print('features  {}'.format(n_mels))
    print('training  {}'.format(i_train))
    print('test:     {}'.format(n_samples - i_train))

    try:
        os.mkdir('./cache/synth/')
    except FileExistsError:
        pass
    save_array(x_test, './cache/synth/waveform_test.npy')
    save_array(y_test, './cache/synth/features_test.npy')
    save_array(x_train, './cache/synth/waveform_train.npy')
    save_array(y_train, './cache/synth/features_train.npy')


if __name__ == '__main__':
    main()
