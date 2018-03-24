import glob

import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError

from . import config
from utils import whole_multiple


def clip_to_slice_multiple(x):
    return x[:whole_multiple(tf.shape(x)[0], config.slice_size), :]


def normalise(x):
    mean, var = tf.nn.moments(x, axes=[0])
    std = tf.sqrt(var)
    return (x - mean) / std


_session = None


def _get_session():
    global _session
    if _session is None:
        _session = tf.Session()
        _session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        try:
            saver.restore(_session, './save/synth')
        except NotFoundError:
            pass
    return _session


_waveform = tf.placeholder(tf.float32, (None, config.channels), 'waveform')
_waveform_to_codings_op = config.model.encoder(
    config.model.prepare(config.model.generate_features(clip_to_slice_multiple(
        normalise(
            _waveform)))
    )
)

_codings = tf.placeholder(tf.float32, (None, config.coding_size), 'codings')
_codings_to_waveform_op = tf.reshape(config.model.data_from_features(
    config.model.decoder(_codings)),
    (-1, config.channels)
)


def waveform_to_codings(waveform):
    return _get_session().run(_waveform_to_codings_op, feed_dict={
        _waveform: waveform,
        **config.model.testing_feeds()
    })


def codings_to_waveform(codings):
    return _get_session().run(_codings_to_waveform_op, feed_dict={
        _codings: codings
    })
