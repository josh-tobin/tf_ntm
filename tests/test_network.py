import sys
from os import path
sys.path.append('/'.join(str.split(path.abspath(__file__), '/')[:-2]))
#from network.input import Input
from network.cell import ConnectorSpecification, Cell
from network.layers import DenseLayer, MLP
from network.network import Network
#from network.dense_layer import DenseLayer
import tensorflow as tf
import numpy as np

'''
def test_input():
    n = np.random.randint(1,100)
    input = Input(n)
    i2 = input
    i3 = Input(n)
    assert(list(input.output_shape) == input.output.get_shape().as_list())
    assert(input == i2)
    assert(input != i3)
'''

def test_connector_spec():
    print 'Testing ConnectorSpecification ...'
    cs1 = ConnectorSpecification([(1,2,3), (1,2,3), (1,2,3)])
    cs2 = ConnectorSpecification([(1,2,3), (1,2,3), (1,2,3)])
    assert cs1 == cs2
    assert cs1.compatible(cs2)
    cs3 = ConnectorSpecification([(1,2,3), (1,2,3)])
    assert cs1 != cs3
    assert not cs1.compatible(cs3)
    cs4 = ConnectorSpecification((1,2,3))
    cs5 = ConnectorSpecification((None, 1, 2, 3))
    assert not cs4.compatible(cs5)
    cs6 = ConnectorSpecification((None, 2, 3))
    assert cs4 != cs6
    assert cs4.compatible(cs6)
    cs7 = ConnectorSpecification(())
    assert cs7.compatible(cs1)
    cs8 = ConnectorSpecification((None, None, 3))
    assert cs8.compatible(cs6)
    print '... Passed!'

def test_cell():
    print 'Testing Cell ...'
    cell = Cell()
    assert cell.input_spec == cell.output_spec
    print '... Passed!'

def test_dense_layer():
    print 'Testing DenseLayer ...'
    # TODO: Verify the output achieves the desired value
    m = np.random.randint(1,100)
    n = np.random.randint(1,100)
    dl = DenseLayer(m, n)
    input = tf.placeholder(tf.float32, [1, m])
    output, _ = dl(input)
    assert output.get_shape()[1] == n
    x = np.random.randn(1, m).astype(np.float32)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    y_ = sess.run(output, feed_dict={input: x})
    assert y_.shape == (1, n)
    print '... Passed!'

def test_MLP():
    print 'Testing MLP ...'
    # TODO: Verify the output achieves the desired value
    # TODO:
    m = np.random.randint(1,100)
    k = np.random.randint(1,100)
    n = np.random.randint(1,100)
    n_layer = np.random.randint(1, 10)
    layer_size = np.random.randint(1, 32)
    mlp = MLP(m, n_layer, layer_size, n)
    mlp2 = MLP(m, n_layer, [layer_size] * n_layer, n)
    input = tf.placeholder(tf.float32, [1,m])
    input2 = tf.placeholder(tf.float32, [1,k])
    mlp3  = MLP([m, k], n_layer, layer_size, n)
    x = np.random.randn(1, m).astype(np.float32)
    x_2 = np.random.randn(1, k).astype(np.float32)
    output, _ = mlp(input)
    output2, _ = mlp2(input)
    output3, _ = mlp3([input, input2])
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    y_ = sess.run(output, feed_dict={input: x})
    assert y_.shape == (1, n)
    y_2 = sess.run(output2, feed_dict={input: x})
    assert y_2.shape == (1, n)
    y_3 = sess.run(output3, feed_dict={input: x, input2: x_2})
    assert y_3.shape == (1, n)
    print '... Passed!'

def test_training():
    print 'Testing training of basic Network ...'
    threshold = 1.0
    batch_size = 5
    mlp = MLP(1, 1, 32, 1)
    optimizer = tf.train.AdamOptimizer()
    x = np.random.randn(50,1)
    y = 2*x + 1
    x_ = tf.placeholder(tf.float32, [batch_size,1])
    loss = lambda a, b: tf.nn.l2_loss(a-b)
    mlp = Network(mlp, x_)
    mlp.compile(loss, optimizer)
    losses = mlp.train(x, y, batch_size=batch_size, n_epochs=200, verbose=False)
    assert losses[-1] < threshold
    print '... Passed!'

TESTS = [
    test_connector_spec,
    test_cell,
    test_dense_layer,
    test_MLP,
    test_training,
    ]

if __name__ == '__main__':
    for test in TESTS:
        test()
    print '---------------------------------------'
    print "Test_network complete. %d tests passed."%len(TESTS)