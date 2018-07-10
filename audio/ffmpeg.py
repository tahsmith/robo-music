import subprocess
import shutil
import numpy as np
import sys

FFMPEG_PATH = shutil.which('ffmpeg')

np_dtype_to_ffmpeg = {
    np.int16: 's16'
}


def make_ffmpeg_format(dtype):
    suffix = 'le' if sys.byteorder == 'little' else 'be'
    if dtype == np.int16:
        return 's16' + suffix
    else:
        raise ValueError('Unsupported dtype: {}'.format(dtype))


def load_sound_file(file_name, dtype, sample_rate, channels):
    out_format = make_ffmpeg_format(dtype)

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
        error = ValueError(ffmpeg_process.stderr.read())
        raise error

    return waveform


def write_sound_file(waveform, file_name, sample_rate):
    input_format = make_ffmpeg_format(waveform.dtype)

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

    ffmpeg_process.stdin.write(waveform.tobytes())

    ffmpeg_process.stdin.close()

    error_code = ffmpeg_process.wait()
    if error_code:
        raise subprocess.CalledProcessError(error_code, ffmpeg_process.args,
                                            stderr=ffmpeg_process.stderr.read())
