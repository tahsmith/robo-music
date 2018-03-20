import tensorflow as tf
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
    return _session


_waveform = tf.placeholder(tf.float32, (None, config.channels))
_waveform_to_codings_op = config.model.encoder(
    config.model.prepare(config.model.preprocess(clip_to_slice_multiple(
        normalise(
            _waveform)))
    )
)


def waveform_to_codings(waveform):
    return _get_session().run(_waveform_to_codings_op, feed_dict={_waveform:
                                                                waveform})
