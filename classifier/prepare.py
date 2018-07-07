import glob

import librosa
import numpy as np
import os

import sys

N_FEATURES = 128
SAMPLE_RATE = 11025
FRAME_SIZE = 2048
FRAME_TIME = FRAME_SIZE / SAMPLE_RATE


def make_features(waveform):
    spec = librosa.feature.melspectrogram(
        waveform.reshape((-1,)),
        SAMPLE_RATE,
        n_fft=FRAME_SIZE,
        hop_length=FRAME_SIZE // 8,
        n_mels=N_FEATURES
    )

    spec = spec.transpose((1, 0))
    return spec


def sample_to_batch(waveform, category):
    features = make_features(waveform)
    n_frames = features.shape[0]
    categories = np.ones((n_frames,), dtype=np.int32) * category
    return features, categories


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
    categories = list_categories()

    for i, category in enumerate(categories):
        print('{}: {}'.format(i, category))

    x = np.empty((0, N_FEATURES), np.float32)
    y = np.empty(0, dtype=np.int32)

    i = 0
    for cat_i, category in enumerate(categories):
        for file in get_samples(category):
            waveform = np.load(file)
            x_batch, y_batch = sample_to_batch(waveform, cat_i)
            x = np.concatenate((x, x_batch))
            y = np.concatenate((y, y_batch))
            print(file)

    n_samples = x.shape[0]
    indices = np.arange(0, n_samples)
    np.random.shuffle(indices)
    x = x[indices, :]
    y = y[indices]

    train_percent = 90
    i_train = (n_samples * train_percent) // 100
    x_train = x[0:i_train]
    y_train = y[0:i_train]

    x_test = x[i_train:]
    y_test = y[i_train:]

    print("Summary:")
    print('features  {}'.format(N_FEATURES))
    print('training  {}'.format(i_train))
    print('test:     {}'.format(n_samples - i_train))

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
