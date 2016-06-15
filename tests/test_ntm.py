import sys
from os import path
sys.path.append('/'.join(str.split(path.abspath(__file__), '/')[:-2]))
from ntm.addressing_mechanism import AddressingCell, ShortcircuitAddressing
from network.layers import MLP, RNNLayer
from ntm.head import Head, ReadHead, WriteHead
from ntm.ntm_cell import NTMCell
from network.network import Network
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell



def test_addressing_cell():
    state_size = 100
    memory_shape = (1000, 50)
    batch_size = 70
    memory_size = memory_shape[0]
    addr = AddressingCell(state_size, memory_shape, batch_size=batch_size)
    assert isinstance(addr, Cell)
    controller_state = tf.placeholder(tf.float32, [batch_size, state_size])
    memory_state = tf.placeholder(tf.float32, [batch_size,] + list(memory_shape))
    prev_w = (1./memory_shape[0])*tf.ones([batch_size, memory_size])

    focus_tensor = addr._focus_content(controller_state, memory_state)
    assert focus_tensor.get_shape()[0] == batch_size
    assert focus_tensor.get_shape()[1] == memory_shape[0]

    interp_tensor = addr._interpolate(controller_state, focus_tensor, prev_w)
    assert interp_tensor.get_shape()[0] == batch_size
    assert interp_tensor.get_shape()[1] == memory_shape[0]

    conv_tensor = addr._convolutional_shift(controller_state, interp_tensor)
    assert conv_tensor.get_shape()[0] == batch_size
    assert conv_tensor.get_shape()[1] == memory_size

    #sharpened_test = addr._sharpen_test(controller_state, memory_state, prev_w)

    sharpened_tensor  = addr._sharpen(controller_state, conv_tensor)
    assert sharpened_tensor.get_shape()[0] == batch_size
    assert sharpened_tensor.get_shape()[1] == memory_size

    addr_tensor, _ = addr([controller_state, memory_state], prev_w)
    assert addr_tensor.get_shape()[0] == batch_size
    assert addr_tensor.get_shape()[1] == memory_size

    random_state = np.random.randn(batch_size, state_size)
    random_memory = np.random.randn(batch_size, memory_shape[0], memory_shape[1])

    eps = 1e-4

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    dict = {controller_state: random_state, memory_state: random_memory}
    focus_output = sess.run(focus_tensor, feed_dict=dict)
    assert (np.abs(np.sum(focus_output, axis=1) - 1.0) < eps).any()
    interp_output = sess.run(interp_tensor, feed_dict=dict)
    conv_output = sess.run(conv_tensor, feed_dict=dict)

    '''
    # DEBUG #

    s0, s1, s2, s3, s4 = sess.run(sharpened_test, feed_dict=dict)
    print "w_c"
    print np.sum(s0, axis=1)
    print "w_g"
    print np.sum(s1, axis=1)
    print "w_hat"
    print np.sum(s2, axis=1)
    print "sharpened (step 1)"
    print np.sum(s3, axis=1)
    print "sharpened (step 2)"
    print np.sum(s4, axis=1)
    '''

    sharp_output = sess.run(sharpened_tensor, feed_dict=dict)
    assert (np.abs(np.sum(sharp_output, axis=1) - 1.0) < eps).any()
    full_output = sess.run(addr_tensor, feed_dict=dict)
    assert (np.abs(np.sum(full_output, axis=1) - 1.0) < eps).any()

def test_head():
    state_size = 100
    memory_shape = (1000, 50)
    batch_size = 70
    head = Head(state_size, memory_shape, batch_size=batch_size)
    read_head = ReadHead(state_size, memory_shape, batch_size=batch_size)
    write_head = WriteHead(state_size, memory_shape, batch_size=batch_size)

    controller_state = tf.placeholder(tf.float32, [batch_size, state_size])
    memory_state = tf.placeholder(tf.float32, [batch_size,] + list(memory_shape))
    prev_w = (1./memory_shape[0])*tf.ones([batch_size, memory_shape[0]])

    read_result, _ = read_head([controller_state, memory_state], prev_w)
    assert read_result.get_shape()[0] == batch_size
    assert read_result.get_shape()[1] == memory_shape[1]

    write_result, _ = write_head(controller_state, memory_state, prev_w)
    assert write_result.get_shape() == memory_state.get_shape()

    random_state = np.random.randn(batch_size, state_size)
    random_memory = np.random.randn(batch_size, memory_shape[0], memory_shape[1])

    eps = 1e-4

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    dict = {controller_state: random_state, memory_state: random_memory}

    read = sess.run(read_result, feed_dict=dict)
    write = sess.run(write_result, feed_dict=dict)

def test_ntm_cell():
    state_size = 10
    memory_shape = (100, 5)
    batch_size = 7
    controller_num_layers = 1
    controller_hidden_size = 32
    input_size = 16
    controller_network = MLP((input_size, memory_shape[1]),
                             controller_num_layers, controller_hidden_size, state_size)

    ntm = NTMCell(controller_network, memory_shape, batch_size)


    input_tensor = tf.placeholder(tf.float32, [batch_size, input_size])
    r_prev = tf.random_normal([batch_size, memory_shape[1]])
    s_prev = tf.random_normal([batch_size, state_size])
    memory_prev = tf.random_normal([batch_size] + list(memory_shape))
    r_w_prev = tf.random_normal([batch_size, memory_shape[0]])
    w_w_prev = tf.random_normal([batch_size, memory_shape[0]])

    ntm_o = ntm(input_tensor, r_prev, s_prev, memory_prev, [r_w_prev],
                             [w_w_prev])
    ntm_out, ntm_state = ntm_o
    assert ntm_out.get_shape()[0] == batch_size
    assert ntm_out.get_shape()[1] == state_size
    assert len(ntm_state) == 5 # r_t, s_t, memory_t, w_r, w_w
    assert ntm_state[0].get_shape() == r_prev.get_shape()
    assert ntm_state[1].get_shape() == s_prev.get_shape()
    assert ntm_state[2].get_shape() == memory_prev.get_shape()
    assert ntm_state[3][0].get_shape() == r_w_prev.get_shape()
    assert ntm_state[4][0].get_shape() == w_w_prev.get_shape()

    random_input = np.random.randn(batch_size, input_size)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    ntm_result = sess.run(ntm_out, feed_dict={input_tensor: random_input})

def test_ntm_layer():
    state_size = 10
    memory_shape = (100, 5)
    batch_size = 7
    controller_num_layers = 1
    controller_hidden_size = 32
    input_size = 16
    T = 9
    controller_network = MLP((input_size, memory_shape[1]),
                             controller_num_layers, controller_hidden_size,
                             state_size)

    x = tf.placeholder(tf.float32, [batch_size, T, input_size])
    ntm_cell = NTMCell(controller_network, memory_shape, batch_size)
    ntm = Network(ntm_cell, x)
    output, final_state = ntm.output()
    assert output.get_shape()[0] == batch_size
    assert output.get_shape()[1] == T
    assert output.get_shape()[2] == state_size

    x_ = np.random.randn(batch_size, T, input_size)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    y_ = sess.run(output, feed_dict={x: x_})
    assert not np.isnan(y_).any()
    assert not np.isinf(y_).any()

def test_rnn():
    input_size = 1
    output_size = 1
    T = 10
    n_batches = 5
    batch_size = 5
    cell = RNNLayer(input_size, output_size, batch_size)
    x = tf.placeholder(tf.float32, [batch_size, T, input_size])
    y = tf.placeholder(tf.float32, [batch_size, T, output_size])
    network = Network(cell, x)
    x_ = np.random.randn(batch_size, T, input_size)
    y_ = 2*x_ + 1
    optimizer = tf.train.GradientDescentOptimizer(1e-4)
    loss = lambda a, b: tf.reduce_mean(tf.pow(a - b, 2))
    #loss = tf.reduce_mean(tf.pow(network.output()[0] - y, 2))
    network.compile(loss, optimizer)
    losses = network.train(x_, y_, batch_size=batch_size, verbose=False)



def test_ntm_gradients():
    state_size = 1
    memory_shape = (5,1)
    batch_size=10
    controller_num_layers=1
    controller_hidden_size=10
    input_size=1
    n_batches=20
    T=2
    controller_network = MLP((input_size, memory_shape[1]),
                             controller_num_layers, controller_hidden_size,
                             state_size)


    x = tf.placeholder(tf.float32, [batch_size, T, input_size])
    x_ = np.random.randn(batch_size*n_batches, T, input_size)
    y_ = 2*x_ + 1.
    addr = ShortcircuitAddressing(memory_shape, batch_size)
    rh = ReadHead(state_size, memory_shape, addresser=addr, batch_size=batch_size,
                  hidden_size=2)
    #ntm_cell = NTMCell(controller_network, memory_shape, batch_size,
    #                   read_head=rh)
    ntm_cell = NTMCell(controller_network, memory_shape, batch_size)
    ntm = Network(ntm_cell, x)
    loss = lambda a, b: tf.nn.l2_loss(a - b)
    optimizer = tf.train.GradientDescentOptimizer(1e-4)
    ntm.compile(loss, optimizer)
    ntm.train(x_, y_, batch_size=batch_size, n_epochs=2)
    #print 'Trainable variables'
    #print len(tf.trainable_variables())
    #print ntm._loss
    #ntm._sess.run(ntm._optimizer.compute_gradients(ntm._loss),
    #              feed_dict={ntm.input: x_[:10, :, :], ntm.y: y_[:10, :, :]})


TESTS = [
    #test_addressing_cell,
    #test_head,
    #test_ntm_cell,
    #test_ntm_layer,
    #test_rnn,
    test_ntm_gradients,
    ]

if __name__ == '__main__':
    for test in TESTS:
        test()
    print "test_network complete. %d tests passed."%len(TESTS)