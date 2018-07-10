import sys

import tensorflow as tf

from synth.model import model_fn, params_from_config


def main(argv):
    model_dir = argv[1]

    from config import config_dict
    slice_size = config_dict['synth']['slice_size']
    channels = config_dict['audio']['channels']
    n_mels = config_dict['classifier']['n_mels']

    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        features={
            'waveform': tf.placeholder(tf.int32,
                                       (None, slice_size, channels),
                                       'waveform'
                                       ),
            'conditioning': tf.placeholder(tf.float32, (None, n_mels),
                                           'conditioning')
        }
    )

    estimator = tf.estimator.Estimator(
        model_dir=argv[1],
        model_fn=model_fn,
        params=params_from_config()
    )

    estimator.export_savedmodel(model_dir, serving_input_fn,)


if __name__ == '__main__':
    main(sys.argv)
