import tensorflow as tf

def cosine_similarity(U, v, eps=1e-7):
    """ Calculates the cosine similarity u.v/(l2(u)*l2(v))
        between all of the vectors in two 3-dimensional tensors
        U and v
    """
    sims_numerator = tf.batch_matmul(U, v)[:, :, 0]
    U_l2 = tf.reduce_sum(tf.pow(U, 2), 2)
    v_l2 = tf.reduce_sum(tf.pow(v, 2), 1)
    sims_denominator = tf.sqrt(tf.maximum(tf.mul(U_l2, v_l2), eps))
    sims = tf.div(sims_numerator, sims_denominator)
    return sims
