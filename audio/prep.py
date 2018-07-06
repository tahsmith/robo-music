import librosa
import numpy as np


def preprocess_waveform(waveform):
    waveform = waveform.astype(np.float32)
    waveform = librosa.effects.trim(waveform)[0]
    waveform_max = np.max(waveform)
    waveform_min = np.min(waveform)

    waveform = (waveform - waveform_min) / (waveform_max - waveform_min)

    return waveform



