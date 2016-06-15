import tensorflow as tf
import numpy as np

def glorot_uniform_initializer(shape):
    # sqrt(6./(shape[0] + shape[1])
    bound = np.sqrt(6./(shape[0] + shape[1]))
    return tf.random_uniform(shape, minval=-bound, maxval=bound)