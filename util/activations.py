import tensorflow as tf

def relu(x, a=1):
    return tf.nn.relu(x) + a