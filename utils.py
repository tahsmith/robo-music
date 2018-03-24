import math

import io
from shutil import which
import numpy as np
import subprocess

FFMPEG_PATH = which('ffmpeg')


def conv_size(input_width, filter_width, stride, padding):
    if padding == 'SAME':
        return math.ceil(float(input_width) / float(stride))
    if padding == 'VALID':
        return math.ceil(float(input_width - filter_width) / float(
            stride)) + 1


def shuffle(*arrays):
    assert (all(x.shape[0] == arrays[0].shape[0] for x in arrays))
    indices = np.arange(0, arrays[0].shape[0])
    np.random.shuffle(indices)
    shuffled = tuple(
        array[indices] for array in arrays
    )
    if len(shuffled) == 1:
        return shuffled[0]
    else:
        return shuffled


def whole_multiple(x, y):
    return x - x % y


def normalise_to_int_range(x, dtype):
    int_info = np.iinfo(dtype)
    max_int = int_info.max
    min_int = int_info.min
    max_val = np.max(x)
    min_val = np.min(x)

    x = (x - min_val) / (max_val - min_val)
    x = x * (max_int - min_int) + min_int
    x = np.rint(x).astype(dtype)
    return x


def load_sound_file(file_name,
                    dtype,
                    sample_rate,
                    channels
                    ):
    if dtype is np.int16:
        out_format = 's16le'
    else:
        raise ValueError('Unsupported dtype: {}'.format(dtype))

    ffmpeg_process = subprocess.Popen(
        [FFMPEG_PATH,
         # '-v', 'quiet',
         '-i', file_name,
         '-f', out_format,
         '-ar', str(sample_rate),
         '-ac', str(channels),
         '-acodec', 'pcm_' + out_format,
         '-'
         ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    waveform = np.frombuffer(ffmpeg_process.stdout.read(), dtype=dtype).reshape(
        (-1, channels)
    )

    error_code = ffmpeg_process.wait()
    if error_code:
        error = subprocess.CalledProcessError(ffmpeg_process.returncode,
                                                  ffmpeg_process.args,
                                                  stderr=ffmpeg_process.stderr.read())
        raise error

    return waveform


def write_sound_file(waveform, file_name, sample_rate):
    if waveform.dtype == np.int16:
        input_format = 's16le'
    else:
        raise ValueError('Unsupported dtype: {}'.format(waveform.dtype))

    channels = waveform.shape[1]
    ffmpeg_process = subprocess.Popen(
        [FFMPEG_PATH,
         '-f', input_format,
         '-ar', str(sample_rate),
         '-ac', str(channels),
         '-acodec', 'pcm_' + input_format,
         '-i', '-',
         '-y',
         file_name
         ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    ffmpeg_process.stdin.write(bytes(waveform))

    ffmpeg_process.stdin.close()

    error_code = ffmpeg_process.wait()
    if error_code:
        raise subprocess.CalledProcessError(error_code, ffmpeg_process.args,
                                            stderr=ffmpeg_process.stderr.read())
