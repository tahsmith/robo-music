import glob

import librosa
import numpy as np
import os

import sys


N_FEATURES = 128


def make_features(waveform, sample_rate):
    spec = librosa.feature.melspectrogram(waveform.reshape((-1,)), sample_rate)
    log_spec = librosa.core.amplitude_to_db(spec, ref=np.max)
    avg_log_spec = np.mean(log_spec, axis=1)
    return avg_log_spec


def list_categories():
    categories_list = []
    samples_path = './cache/samples'
    for root, dirs, files in os.walk(samples_path):
        if root != samples_path:
            break
        for dir in dirs:
            if dir[0] != '_':
                categories_list.append(dir)
    return categories_list


def get_samples(category):
    return glob.glob('./cache/samples/{}/*.npy'.format(category))


def save_array(x, filename):
    np.save(filename, x, allow_pickle=False)


def main(argv):
    sample_rate = 11025
    categories = list_categories()

    n = 0
    for i, category in enumerate(categories):
        print('{}: {}'.format(i, category))
        n += len(get_samples(category))

    x = np.empty((n, N_FEATURES), np.float32)
    y = np.empty(n, dtype=np.int32)

    i = 0
    for cat_i, category in enumerate(categories):
        for file in get_samples(category):
            waveform = np.load(file)
            x[i, :] = make_features(waveform, sample_rate)
            y[i] = cat_i
            i += 1
            print(file)
    
    n_samples = x.shape[0]
    indices = np.arange(0, n_samples)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    train_percent = 90
    i_train = (n_samples * train_percent) // 100
    x_train = x[0:i_train]
    y_train = y[0:i_train]

    x_test = x[i_train:]
    y_test = y[i_train:]
    
    try:
        os.mkdir('./cache/classifier/')
    except FileExistsError:
        pass
    save_array(x_test, './cache/classifier/x_test.npy')
    save_array(y_test, './cache/classifier/y_test.npy')
    save_array(x_train, './cache/classifier/x_train.npy')
    save_array(y_train, './cache/classifier/y_train.npy')


if __name__ == '__main__':
    main(sys.argv)
