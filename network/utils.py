import tensorflow as tf
import numpy as np

def slice_rnn_input(input_tensor):
    """ Takes 3d input of shape (batch_size, T, state_size)
        and slices it to create a list of 2d inputs [x_0, ..., x_T],
        where each x_i has shape (batch_size, state_size).
    """
    T = input_tensor.get_shape()[1]
    list_of_tensors = tf.split(1, T, input_tensor)
    list_of_tensors = [t[:,0,:] for t in list_of_tensors]
    return list_of_tensors

def combine_rnn_output(outputs):
    """ Takes a list of T tensors of shape (batch_size, output_size)
        and combines them to form a single 3d tensor of shape
        (batch_size, T, output_size)
    """
    reshaped_outputs = [tf.expand_dims(output_, 1) for output_ in outputs]
    combined_output = tf.concat(1, reshaped_outputs)
    return combined_output

def shuffle_data(x, y):
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    return x, y