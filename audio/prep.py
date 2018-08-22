import librosa
import numpy as np


def preprocess_waveform(waveform):
    waveform = waveform.astype(np.float32)
    waveform = librosa.effects.trim(waveform)[0]
    waveform_max = np.max(waveform)
    waveform_min = np.min(waveform)

    waveform = (waveform - waveform_min) / (waveform_max - waveform_min)

    return waveform


def waveform_to_slices(waveform, slice_size, slice_stride):
    samples, channels = waveform.shape
    slice_count = (samples - slice_size) // slice_stride + 1
    slices = np.empty((slice_count, slice_size, channels))
    for i in range(slice_count):
        begin = i * slice_stride
        end = begin + slice_size
        slices[i, :, :] = waveform[begin:end, :]

    return slices
