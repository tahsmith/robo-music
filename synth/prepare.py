import glob
import os
from multiprocessing import Pool
from random import shuffle

from numpy import random

import librosa
import numpy as np

from utils import normalise_to_int_range


def save_array(x, filename):
    np.save(filename, x, allow_pickle=False)


def list_input_files(cache_dir):
    return glob.glob(f'{cache_dir}/samples/**/*.npy', recursive=True) \
           + glob.glob(f'{cache_dir}/music/**/*.npy')


def quantise(x, quantisation):
    if quantisation != 256:
        raise NotImplementedError('quantisation != 256')
    return normalise_to_int_range(x, np.uint8).astype(np.int32)


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

    spec = librosa.power_to_db(spec)

    spec = spec.transpose((1, 0)).astype(np.float32)

    return spec


def augment_samples(sample):
    sample = sample.astype(np.float32)
    scale = np.exp(np.random.randn(sample.shape[0], 1, 1))
    noise = np.random.randn(*sample.shape) * 0.1

    sample *= scale
    sample += noise

    return sample


def augment_data_set(waveform, conditioning, times=2):
    augmented_waveform = waveform
    augmented_conditioning = conditioning
    for i in range(times):
        augmented_waveform = np.concatenate([
            augmented_waveform,
            augment_samples(waveform)
        ], axis=0)

        augmented_conditioning = np.concatenate([
            augmented_conditioning,
            conditioning
        ], axis=0)

    return augmented_waveform, augmented_conditioning


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
    return samples


def clip_to_slice_size(slice_size, waveform):
    remainder = waveform.shape[0] % slice_size
    if remainder != 0:
        waveform = waveform[:waveform.shape[0] - remainder, :]
    return waveform


def pad_waveform(waveform, padding, channels):
    left_pad = padding // 2
    right_pad = padding // 2 + padding % 2

    return np.concatenate([
        np.zeros((left_pad, channels), dtype=np.float32),
        waveform,
        np.zeros((right_pad, channels), dtype=np.float32)
    ])


def normalise_waveform(waveform, channels):
    waveform = np.array(waveform, dtype=np.float32)
    waveform = np.reshape(waveform, (-1, channels))
    waveform = (waveform - np.mean(waveform)) / np.std(waveform)
    return waveform


def files_to_waveform_chunks(file_list, channels, chunk_size, slice_size):
    chunk = np.zeros((0, channels))
    file_list = iter(enumerate(file_list))
    i = -1
    current_file_data = np.zeros((0, channels))
    while True:
        try:
            if current_file_data.shape[0] == 0:
                i, file = next(file_list)
                current_file_data = np.load(file)
                current_file_data = normalise_waveform(current_file_data,
                                                       channels)
                current_file_data = pad_waveform(current_file_data, slice_size,
                                                 channels)

            remaining_space = chunk_size - chunk.shape[0]
            if remaining_space > 0:
                cut = min(current_file_data.shape[0], remaining_space)
                chunk = np.concatenate([
                    chunk,
                    current_file_data[:cut, :]
                ])
                current_file_data = current_file_data[cut:, :]
            else:
                yield i, chunk
                chunk = np.zeros((0, channels))
        except StopIteration:
            # yield i, chunk
            break


def waveform_to_samples(waveform, sample_rate, slice_size,
                        stride, n_mels, quantisation, augmentation):
    features = compute_features(waveform, sample_rate, slice_size,
                                stride, n_mels)

    # last slice may not be chunck sized
    waveform = clip_to_slice_size(slice_size, waveform)
    samples = create_samples(waveform, slice_size, stride)
    samples, features = augment_data_set(samples, features, augmentation)
    samples = quantise(samples, quantisation)

    return samples, features


def file_shuffle(file_count):
    choices = list(range(file_count))
    choice1, choice2 = random.choice(choices, size=2, replace=False)
    waveform1 = np.load(f'./cache/synth/waveform_{choice1}.npy')
    waveform2 = np.load(f'./cache/synth/waveform_{choice2}.npy')
    features1 = np.load(f'./cache/synth/features_{choice1}.npy')
    features2 = np.load(f'./cache/synth/features_{choice2}.npy')

    size = waveform1.shape[0]
    assert waveform2.shape[0] == size

    waveform_all = np.concatenate([waveform1, waveform2])
    features_all = np.concatenate([features1, features2])

    indices = np.arange(0, size * 2)
    np.random.shuffle(indices)
    waveform_all = waveform_all[indices, :, :]
    features_all = features_all[indices, :]

    waveform1 = waveform_all[:size, :, :]
    features1 = features_all[:size, :]

    waveform2 = waveform_all[size:, :, :]
    features2 = features_all[size:, :]

    save_array(waveform1, f'./cache/synth/waveform_{choice1}.npy')
    save_array(waveform2, f'./cache/synth/waveform_{choice2}.npy')
    save_array(features1, f'./cache/synth/features_{choice1}.npy')
    save_array(features2, f'./cache/synth/features_{choice2}.npy')


def main():
    from config import config_dict
    synth_config = config_dict['synth']
    audio_config = config_dict['audio']
    data_config = config_dict['data']

    slice_size = synth_config['slice_size']
    quantisation = synth_config['quantisation']
    channels = audio_config['channels']
    sample_rate = audio_config['sample_rate']
    n_mels = config_dict['classifier']['n_mels']
    stride = synth_config['sample_stride']
    augmentation = synth_config['sample_augmentation']

    input_files = list_input_files(data_config['cache'])
    shuffle(input_files)

    chunck_size = 400000  # this value comes from librosa.stft
    chunck_size = chunck_size - chunck_size % slice_size

    def generate_waveforms():
        return files_to_waveform_chunks(
            input_files,
            channels,
            chunck_size,
            slice_size
        )

    def log_progress(tup):
        i, x = tup
        print(f'file {i} / {len(input_files)}: {input_files[i]}')
        return x

    def make_batch(x):
        return waveform_to_samples(log_progress(x), sample_rate,
                                   slice_size, stride, n_mels,
                                   quantisation, augmentation)

    samples = (make_batch(x) for x in generate_waveforms())

    try:
        os.mkdir('./cache/synth/')
    except FileExistsError:
        pass

    n_samples = 0

    for i, (samples, features) in enumerate(samples):
        save_array(samples, f'./cache/synth/waveform_{i}.npy')
        save_array(features, f'./cache/synth/features_{i}.npy')
        n_samples += samples.shape[0]

    print("Summary:")
    print('features  {}'.format(n_mels))
    print('samples   {}'.format(n_samples))

    for n in range(2 * i):
        print('shuffling')
        file_shuffle(i)


if __name__ == '__main__':
    main()
