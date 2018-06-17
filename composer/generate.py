import tensorflow as tf

from utils import write_sound_file, normalise_to_int_range

synth_graph = tf.Graph()
with synth_graph.as_default():
    from synth.transform import codings_to_waveform
import numpy as np
from . import config
import synth.config
from . import model


def generate():
    batch = np.random.uniform(0.0, 0.1, (1, config.n_steps, config.n_inputs))
    wave_form = np.zeros((0, config.n_inputs))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './save/composer')
        for i in range(120 * synth.config.samples_per_second //
                       synth.config.slice_size):
            new_point = sess.run(model.outputs,
                                 feed_dict={model.codings: batch})
            batch = np.concatenate((batch[:, 1:, :], new_point[:, -1:, :]),
                                   axis=1)
            new_point = new_point[0, -1, :].reshape((1, config.n_inputs))
            wave_form = np.concatenate((wave_form, new_point))

            length = (i * synth.config.slice_size /
                      synth.config.samples_per_second)
            print(f'{length}s')

    with synth_graph.as_default():
        wave_form = codings_to_waveform(wave_form)
    wave_form = normalise_to_int_range(wave_form, np.int16)
    print(wave_form.shape)
    write_sound_file(wave_form, 'generated.mp3', synth.config.samples_per_second)
    return wave_form


if __name__ == '__main__':
    generate()