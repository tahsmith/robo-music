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


def walk_dir(sample_rate, channels, data_dir, cache_dir):
    for root, dirs, files in os.walk(data_dir):
        cache_root = root.replace(data_dir, cache_dir)
        if not os.path.isdir(cache_root):
            os.mkdir(cache_root)
        for file in files:
            if file[0] not in {'.', '_'}:
                file_path = root + '/' + file
                prep_file(file_path,
                          cache_root + '/' + file,
                          sample_rate,
                          channels)


def main(argv):
    from config import parse_config
    config_dict = parse_config('./config.yml')
    audio_dict = config_dict['audio']
    data_config = config_dict['data']
    os.makedirs(data_config['cache'], exist_ok=True)

    walk_dir(audio_dict['sample_rate'], audio_dict['channels'],
             data_config['input'], data_config['cache'])


if __name__ == '__main__':
    main(sys.argv)
