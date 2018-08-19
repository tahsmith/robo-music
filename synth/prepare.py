import glob
import os
from functools import partial
from random import shuffle
import asyncio
from concurrent.futures import ProcessPoolExecutor

from numpy import random

import librosa
import numpy as np

from synth.model import params_from_config


def save_array(x, filename):
    np.save(filename, x, allow_pickle=False)


def list_input_files(cache_dir):
    return glob.glob(f'{cache_dir}/samples/**/*.npy', recursive=True) \
           + glob.glob(f'{cache_dir}/music/*.npy')


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    mu = np.float32(quantization_channels - 1.0)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    mu_law = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    return np.int32((mu_law + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * (np.float32(output) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)
    return np.sign(signal) * magnitude


def quantise(x, quantisation):
    if quantisation != 256:
        raise NotImplementedError('quantisation != 256')
    return mu_law_encode(x, quantisation)


def compute_features(waveform, sample_rate, feature_window, n_mels):
    waveform_padded = pad_waveform(waveform, feature_window - 1, 0, 0)

    if waveform_padded.shape[1] == 1:
        waveform_padded = waveform_padded.reshape((-1))
    spec = np.abs(
        librosa.stft(waveform_padded, n_fft=feature_window, hop_length=1,
                     center=False)) ** 2
    spec = librosa.feature.melspectrogram(
        waveform,
        sample_rate,
        S=spec,
        n_mels=n_mels
    )
    spec = librosa.power_to_db(spec)

    spec = spec.transpose((1, 0)).astype(np.float32)

    assert spec.shape[0] == waveform.shape[0]

    return spec


def augment_sample(sample, noise_level=0.0, scale_range=1.0):
    sample = sample.astype(np.float32)
    scale = scale_range ** np.random.uniform(-1.0, 1.0)
    noise = np.random.randn(*sample.shape) * noise_level

    sample *= scale
    sample += noise

    return sample


def augment(waveform, times=2, noise=0.0, scale=1.0):
    augmented_waveform = waveform
    for i in range(times):
        augmented_waveform = np.concatenate([
            augmented_waveform,
            augment_sample(waveform, noise, scale)
        ], axis=0)

    return augmented_waveform


def clip_to_slice_size(slice_size, waveform):
    remainder = waveform.shape[0] % slice_size
    if remainder != 0:
        waveform = waveform[:waveform.shape[0] - remainder, :]
    return waveform


def pad_waveform(waveform, left_pad, right_pad, noise=0.0):
    channels = waveform.shape[1]
    return np.concatenate([
        np.random.randn(left_pad, channels) * noise,
        waveform,
        np.random.randn(right_pad, channels) * noise
    ])


def normalise_waveform(waveform, channels):
    waveform = np.array(waveform, dtype=np.float32)
    waveform = np.reshape(waveform, (-1, channels))
    waveform = waveform - np.mean(waveform)
    waveform_range = np.max(np.abs(waveform))
    waveform /= waveform_range

    assert np.all(waveform <= 1.0)
    assert np.all(waveform >= -1.0)

    return waveform


def files_to_waveform_chunks(file_list, channels, chunk_size, receptive_field,
                             augmentation, noise, scale):
    chunk = np.zeros((0, channels))
    file_list = iter(enumerate(file_list))
    i = -1
    current_file_data = np.zeros((0, channels))
    while True:
        try:
            if current_file_data.shape[0] == 0:
                i, file = next(file_list)
                current_file_data = np.load(file)
                current_file_data = pad_waveform(
                    current_file_data,
                    receptive_field - 1,
                    0,
                    noise
                )
                current_file_data = augment(current_file_data, augmentation,
                                            noise, scale)
                current_file_data = normalise_waveform(current_file_data,
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
                assert chunk.shape[0] == chunk_size
                yield chunk
                chunk = np.zeros((0, channels))
        except StopIteration:
            # yield chunk
            break


def waveform_to_samples(waveform, sample_rate, slice_size, feature_window,
                        n_mels, quantisation):
    waveform = clip_to_slice_size(slice_size, waveform)
    features = compute_features(waveform, sample_rate, feature_window, n_mels)

    waveform = quantise(waveform, quantisation)

    waveform = waveform.reshape(-1, slice_size, waveform.shape[1])
    features = features.reshape(-1, slice_size, n_mels)

    assert np.all(waveform <= 255)
    assert np.all(waveform >= 0)

    return waveform, features


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


async def make_batch(output_path, sample_rate, slice_size, feature_window,
                     n_mels, quantisation, i, x):
    slices, features = await asyncio.get_event_loop().run_in_executor(
        None,
        waveform_to_samples,
        x, sample_rate, slice_size, feature_window, n_mels,
        quantisation
    )

    save_array(slices, f'{output_path}/synth/waveform_{i}.npy')
    save_array(features, f'{output_path}/synth/features_{i}.npy')

    print(f'batch {i}: {slices.shape} {features.shape}')
    return slices.shape[0]


async def main():
    from config import config_dict
    synth_config = config_dict['synth']
    audio_config = config_dict['audio']
    data_config = config_dict['data']

    feature_window = synth_config['feature_window']
    slice_size = synth_config['slice_size']
    quantisation = synth_config['quantisation']
    channels = audio_config['channels']
    sample_rate = audio_config['sample_rate']
    n_mels = config_dict['classifier']['n_mels']
    augmentation = synth_config['sample_augmentation']
    augmentation_noise = synth_config['augmentation_noise']
    augmentation_scale_range = synth_config['augmentation_scale_range']

    model_params = params_from_config()

    cache_path = data_config['cache']
    input_files = list_input_files(cache_path)
    print(f'files: {input_files}')
    shuffle(input_files)

    chunk_size = synth_config['prepare_batch_size']  # 400000 is the max. This
    # value comes from librosa.stft

    chunk_size = chunk_size - chunk_size % slice_size

    def generate_waveforms():
        return files_to_waveform_chunks(input_files, channels, chunk_size,
                                        model_params.receptive_field,
                                        augmentation,
                                        augmentation_noise,
                                        augmentation_scale_range)

    try:
        os.makedirs(f'{cache_path}/synth/', exist_ok=True)
    except FileExistsError:
        pass

    make_batch_part = partial(make_batch, cache_path, sample_rate,
                              slice_size, feature_window, n_mels, quantisation)
    executor = ProcessPoolExecutor(int(config_dict['sys']['cpus']))
    loop = asyncio.get_event_loop()
    loop.set_default_executor(executor)

    processed = asyncio.gather(*(
        make_batch_part(i, x)
        for i, x in enumerate(generate_waveforms())
    ))

    n_samples = sum(await processed)

    print("Summary:")
    print('features  {}'.format(n_mels))
    print('samples   {}'.format(n_samples))


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
