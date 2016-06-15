"""
Several simple neural network layers, implemented as Cell objects.
"""
import numpy as np
from cell import Cell, ConnectorSpecification
import tensorflow as tf
from initializers import glorot_uniform_initializer



class IdentityLayer(Cell):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def __call__(self, x, *state):
        return x, state

class RNNLayer(Cell):
    def __init__(self, input_dim, num_units, batch_size,
                 initializer=glorot_uniform_initializer,
                 activation=tf.nn.relu):
        self._W = tf.Variable(initializer([input_dim, num_units]))
        self._b = tf.Variable(tf.zeros([num_units,]))
        self._batch_size=batch_size
        self._activation = activation
        self._num_units = num_units

    def __call__(self, x, *state):
        x =  self._activation(tf.matmul(x, self._W) + self._b)
        return x, [x]

    def default_state(self):
        return [tf.zeros([self._batch_size,self._num_units])]

class DenseLayer(Cell):

    def __init__(self, input_shape, num_units, initializer=glorot_uniform_initializer,
                 activation=tf.nn.relu):
        """
        Dense (feedforward) neural network layer. Given weights W and bias b,
        maps an input tensor x to activation(np.dot(x, W) + b)
        :param input_shape: expected dimensionality of the input tensor
        :param num_units: dimensionality of the result (i.e., number of cols of W)
        :param initializer: initialization function for the weights W and biases b
        :param activation: activation function (nonlinearity)
        """
        super(DenseLayer, self).__init__()
        if isinstance(input_shape, int):
            self._input_shape = (input_shape,)
        else:
            self._input_shape = input_shape
        self._num_units = num_units
        self._W = tf.Variable(initializer([self._input_shape[-1], self._num_units]),
                              trainable=True)
        self._b = tf.Variable(tf.zeros([num_units,]),
                              trainable=True)
        self._input_spec = ConnectorSpecification([(None, input_shape)])
        self._output_spec = ConnectorSpecification([(None, num_units)])
        self._activation = activation

    @property
    def weights(self):
        return [self._W, self._b]

    def __call__(self, x, *state):
        return self._activation(tf.matmul(x, self._W) + self._b), []

class MergeLayer(Cell):
    def __init__(self, mode='concat', axis=-1):
        """
        Cell that merges several input tensors into a single output tensor.
        Currently supports two modes: 'concat' combines the tensors along
        the given axis, and sum adds them componentwise.
        :param mode: method of merging. Currently supports 'concat' and 'sum'
        :param axis: concatenation axis for mode='concat'
        """
        self._mode = mode
        self._axis = axis

    def __call__(self, x, *state):
        if self._mode == 'concat':
            return tf.concat(self._axis, x), []

        elif self._mode == 'sum':
            return tf.add_n(x), []

        else:
            raise Exception('Unrecognized mode for MergeLayer')

class Conv1dLayer(Cell):
    def __init__(self, input_size, filter_size, stride=1, padding='SAME',
                 initializer=tf.random_normal):
        """
        Performs a 1-d convolution on a batch of input vectors. The filter for
        the convolution is parameterized by some filter weights that are stored
        in self._filter.
        Note: depth dimension not yet supported.

        :param input_size: int representing dimensionality of input
        :param filter_size: int representing the size of the filter
        :param stride: int, stride of the convolution
        :param padding: see tf.nn.conv2d
        :param initializer: initialization function for the filter weights
        """
        self._input_size = input_size
        self._filter_size = filter_size
        self._stride = stride
        self._padding = padding
        self._filter = tf.Variable(initializer([1, filter_size, 1, 1]))

    def __call__(self, x, *state):
        reshaped_input = tf.reshape(x, [-1, 1, self._input_size, 1])
        batch_size = reshaped_input.get_shape().dims[0]
        conv = tf.nn.conv2d(reshaped_input, self._filter, strides=[1,1,self._stride,1],
                            padding=self._padding)
        return tf.reshape(conv, [batch_size, -1]), []

class BatchedConv1dLayer(Cell):
    def __init__(self, input_size, batch_size, stride=1, padding='SAME'):
        """
        Similarly to Conv1dLayer, this layer performs a 1-d convolution on a
        batch of input vectors. However, instead of parameterizing the filter(s)
        by a weight tensor as in Conv1dLayer, BatchedConv1dLayer expects a second
        input filter_tensor that parameterizes the filter(s) as, e.g., the output
        of some other neural network. The primary implementation difference
        is that the filters for BatchedConv1dLayer are not assumed to be the same
        for every input in the batch.
        Note: depth dimension not yet supported.

        :param input_size: int, the dimensionality of the input
        :param batch_size: int, the total inputs in each batch
        :param stride: int, stride of the convolution
        :param padding: see tf.nn.conv2d
        """
        self._input_size = input_size
        self._stride = stride
        self._padding = padding
        self._batch_size = batch_size

    def __call__(self, x, *state):
        input_tensor = x[0]
        reshaped_input = tf.reshape(input_tensor,
                                    [self._batch_size, 1, self._input_size, 1])
        filter_tensor = x[1]
        reshaped_filter = tf.reshape(filter_tensor, [self._batch_size, -1, 1, 1])
        split_input = tf.split(0, self._batch_size, reshaped_input)
        split_filter = tf.split(0, self._batch_size, reshaped_filter)

        filter_applied = []
        for input_, filter in zip(split_input, split_filter):
            filter_applied.append(tf.nn.conv2d(input_, filter,
                                               [1, 1, 1, 1], self._padding))
        return tf.concat(0, filter_applied)[:,0,:,0], []

class MLP(Cell):
    def __init__(self, input_shape, num_hidden, hidden_size, output_shape,
                 initializer=tf.random_normal,
                 hidden_activation=tf.nn.relu, output_activation=tf.identity):
        """
        Basic feedforward multilayer perceptron network. Accepts a (list of)
        tensor(s) and returns a tensor corresponding to the output of the network.
        Allows passing several input tensors, which will be concatenated before
        being fed forward.
        :param input_shape: shape(s) of the inputs. Either an int or a list of
                            ints corresponding to each of the layers
        :param num_hidden: number of hidden layers
        :param hidden_size: size of hidden layers - single int or list of ints
        :param output_shape: dimensionality of output
        :param initializer: initialization function for the weights
        :param hidden_activation: nonlinearity for the hidden layers
        :param output_activation: nonlinearity for the output layer
        """
        if not isinstance(input_shape, list):
            self._input_shape = [input_shape]
        else:
            self._input_shape = input_shape
        assert isinstance(num_hidden, int) and num_hidden > 0
        self._num_hidden = num_hidden
        self._hidden_size = hidden_size
        assert isinstance(self._hidden_size, int) \
                       or (isinstance(self._hidden_size, list)
                           and len(self._hidden_size) == num_hidden)
        if isinstance(self._hidden_size, int):
            self._hidden_size = [self._hidden_size] * self._num_hidden
        self._output_shape = output_shape
        self._initializer = initializer
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation
        self._build_layers()

    @property
    def input_shape(self): return self._input_shape

    @property
    def output_shape(self): return self._output_shape

    def _build_layers(self):
        self._layers = []
        self._layers.append(MergeLayer(axis=1))
        self._layers.append(
            DenseLayer(np.sum(self._input_shape), self._hidden_size[0],
                       initializer=self._initializer,
                       activation=self._hidden_activation))
        for i in range(self._num_hidden - 1):
            self._layers.append(DenseLayer(self._hidden_size[i], self._hidden_size[i+1],
                                           initializer=self._initializer,
                                           activation=self._hidden_activation))
        self._layers.append(DenseLayer(self._hidden_size[-1], self._output_shape,
                                       initializer=self._initializer,
                                       activation=self._output_activation))

    def __call__(self, x, *state):
        out = x
        for layer in self._layers:
            out = layer(out)[0]

        return out, []