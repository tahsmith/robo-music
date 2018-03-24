import synth.config
import math

steps_seconds = 2.0
n_steps = math.ceil(steps_seconds * synth.config.samples_per_second /
                    synth.config.slice_size)
n_inputs = synth.config.coding_size
n_neurons = 50
n_outputs = n_inputs

