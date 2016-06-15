import sys
from os import path
sys.path.append('/'.join(str.split(path.abspath(__file__), '/')[:-2]))
from util.similarity_metrics import cosine_similarity
from network.layers import DenseLayer, BatchedConv1dLayer, MLP
from util import activations
import tensorflow as tf
from network.cell import Cell


class AddressingCell(Cell):

    def __init__(self, state_size, memory_shape, sim=cosine_similarity,
                 hidden_size=32, convolution_range=(-1,0,1), batch_size=32):
        super(AddressingCell, self).__init__()
        self._memory_size = memory_shape[0]
        self._memory_dim = memory_shape[1]
        self._sim = sim
        self._convolution_range = convolution_range
        self._conv_filter_size = len(self._convolution_range)
        self._batch_size = batch_size

        num_hidden = 1
        # _key_network maps controller output -> memory key
        self._key_network = MLP(state_size, num_hidden, hidden_size, self._memory_dim,
                                output_activation=tf.identity)
        # _key_strength_network maps output -> key strength, which is used
        # to scale the memory key
        self._key_strength_network = MLP(state_size, num_hidden, hidden_size,
                                         self._memory_size,
                                         output_activation=tf.nn.relu)
        # _interpolation_gate_network maps controller output -> interpolation
        # factor in [0, 1], which controlls how much of the new address to include
        self._interpolation_gate_network = MLP(state_size, num_hidden, hidden_size,
                                               1, output_activation=tf.sigmoid)
        # _conv_filter network maps controller output -> a 1-d conv filter
        # of width _conv_filter_size which is used to `shift' the weight
        # vector.
        self._conv_filter_network = MLP(state_size, num_hidden, hidden_size,
                                        self._conv_filter_size,
                                        output_activation=tf.nn.softmax)
        self._conv_op = BatchedConv1dLayer(self._memory_size, self._batch_size, stride=1,
                                           padding='SAME')
        # _sharpening_network maps controller output -> sharpening factor
        # in [1, \inf), which `sharpens' the focus on the largest weights
        self._sharpening_network = MLP(state_size, num_hidden, hidden_size,
                                       1, output_activation=activations.relu)

    def get_address(self, controller_state, memory, prev_w):
        """
        Main operation of the addressing mechanism.
        :param controller_state: output of the controller at current timestep
        :param memory: memory at the previous timestep
        :param prev_w: previous weights
        :return: w, the addressing weights at the current timestep
        """
        w_c = self._focus_content(controller_state, memory)
        w_g = self._interpolate(controller_state, w_c, prev_w)
        w_hat = self._convolutional_shift(controller_state, w_g)
        w = self._sharpen(controller_state, w_hat)
        return w, [w]

    def default_state(self):
        # By default, assume uniform probability over memory locations
        return [tf.Variable((1./self._memory_size)
                            * tf.ones([self._batch_size, self._memory_size]),
                            trainable=False)]

    def _focus_content(self, controller_state, memory):
        #TODO: Factor out similarity
        """
        Content-based addressing mechanism

        :param memory: tensor of shape (batch_size x memory_size x memory_dim)
        :return: w_c, a (batch_size x memory_size) vector
        """

        key, _ = self._key_network(controller_state)
        key = tf.expand_dims(key, -1)
        sims = self._sim(memory, key)
        key_strength, _ = self._key_strength_network(controller_state)
        w_c = tf.exp(tf.mul(key_strength, sims))
        w_c = w_c / tf.expand_dims(tf.reduce_sum(w_c, 1), -1)
        return w_c

    def _interpolate(self, controller_state, w_c, prev_w):
        """
        Interpolate between the content-based address and the previous weights.

        :param controller_state: output of the controller at the current timestep
        :param w_c: weights provided by content-based focusing mechanism
        :param prev_w: previous weights output by controller
        :return: w_g, a (batch_size x memory_size) vector
        """
        g_t, _ = self._interpolation_gate_network(controller_state)
        w_g = tf.mul(g_t, w_c) + tf.mul(1-g_t, prev_w)
        return w_g

    def _convolutional_shift(self, controller_state, w_g):
        """
        Perform a 1-d convolutional shift of the interpolated address

        :param controller_state: output of the controller at the current timestep
        :param w_g: interpolated weights
        :return: w_hat, a (batch_size x memory_size) vector
        """
        filters, _ = self._conv_filter_network(controller_state)
        w_hat, _ = self._conv_op([w_g, filters])
        return w_hat

    def _sharpen(self, controller_state, w_hat):
        """
        Sharpen the shifted vector to concentrate weights on a smaller number
        of memory locations

        :param controller_state: output of the controller at the current timestep
        :param w_hat: weights after convolutional shift
        :return: sharpened, the final addressing weights
        """
        # Scale by self._memory_size to prevent underflow
        eps = 1e-6
        w_hat_scaled = self._memory_size * \
                       (w_hat / tf.expand_dims(tf.reduce_sum(w_hat, 1), -1))
        sharpened =  tf.pow(w_hat_scaled,
                            self._sharpening_network(controller_state)[0])
        sharpened = sharpened / tf.expand_dims(tf.reduce_sum(sharpened, 1), -1)
        return sharpened

    def __call__(self, x, *state):
        """
        :param x: Array consisting of [controller_state, memory]
        :param state: Recurrent connections: [prev_w]
        :return: weight vector w
        """
        return self.get_address(x[0], x[1], state[0])

class ShortcircuitAddressing(Cell):
    def __init__(self, memory_size, batch_size):
        #self.weights = tf.Variable(tf.zeros([batch_size, memory_size[0]]),
        #                      trainable=False)
        self.weights = tf.Variable(tf.zeros([memory_size[0],]),
                                   trainable=False)
    def __call__(self, x, *state):
        return self.weights, [self.weights]

    def default_state(self): return self.weights

