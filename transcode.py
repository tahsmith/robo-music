import os
import sys

import numpy as np

from audio.prep import preprocess_waveform
from audio.ffmpeg import load_sound_file


def prep_file(infile, outfile, sample_rate, channels):
    waveform = load_sound_file(infile, np.int16, sample_rate, channels)
    processed_waveform = preprocess_waveform(waveform.astype(np.float32))
    percent_lost = 100 * (1 - processed_waveform.shape[0] / waveform.shape[0])
    print('{file_path}: clipped {percent}%'.format(
        file_path=infile,
        percent=percent_lost
    ))
    np.save(outfile, processed_waveform, allow_pickle=False)


def walk_dir(sample_rate, channels):
    for root, dirs, files in os.walk('./data'):
        cache_dir = root.replace('./data/', './cache/')
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)
        for file in files:
            if file[0] not in {'.', '_'}:
                file_path = root + '/' + file
                prep_file(file_path,
                          cache_dir + '/' + file,
                          sample_rate,
                          channels)


def main(argv):
    from config import parse_config
    audio_dict = parse_config('./config.yml')['audio']

    walk_dir(audio_dict['sample_rate'], audio_dict['channels'])


if __name__ == '__main__':
    main(sys.argv)
