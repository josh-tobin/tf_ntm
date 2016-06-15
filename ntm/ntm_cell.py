import sys
from os import path
import tensorflow as tf

sys.path.append('/'.join(str.split(path.abspath(__file__), '/')[:-2]))
from ntm.head import ReadHead
from ntm.head import WriteHead
from network.cell import Cell
from network.layers import IdentityLayer

"""
TODO
    - Documentation
"""

class NTMCell(Cell):
    def __init__(self, controller_network, memory_size, batch_size,
                 read_head='default', write_head='default', output_net='default',
                 head_hidden_size=32):
        """
        NTMCell represents a single timestep of a Neural Turing Machine.

        :param controller_network: The controller maps the input x_t, previous
                                   read vector r_tm1, and optionally previous
                                   controller statem s_tm1 to a state s_t
        :param memory_size:
        :param batch_size:
        :param read_head:
        :param write_head:
        :param output_net:
        """
        super(NTMCell, self).__init__()
        self._batch_size = batch_size
        self._controller_network = controller_network
        if read_head == 'default':
            self._read_head = ReadHead(controller_network.output_shape, memory_size,
                                       batch_size=batch_size,
                                       hidden_size=head_hidden_size)
        else:
            self._read_head = read_head
        if write_head == 'default':
            self._write_head = WriteHead(controller_network.output_shape, memory_size,
                                         batch_size=batch_size,
                                         hidden_size=head_hidden_size)
        else:
            self._write_head = write_head
        if output_net == 'default':
            self._output_net = IdentityLayer()
        else:
            self._output_net = output_net
        self._memory_size = memory_size
        self._check_input()

    def default_state(self):
        r_0 = tf.Variable(tf.zeros([self._batch_size, self._memory_size[1]]),
                          trainable=False)
        if self._controller_network.default_state():
            s_0 = self._controller_network.default_state()
        else:
            s_0 = tf.Variable(tf.zeros([self._batch_size,
                            self._controller_network.output_shape]),
                              trainable=False)
        memory_0 = tf.Variable(tf.zeros([self._batch_size]
                                        + list(self._memory_size)),
                               trainable=False)
        state = [r_0, s_0, memory_0, self._read_head.default_state(),
                 self._write_head.default_state()]
        return state

    def _check_input(self):
        assert isinstance(self._controller_network, Cell)
        assert isinstance(self._read_head, ReadHead)
        assert isinstance(self._write_head, WriteHead)
        # The controller network should be compatible with the heads
        #assert self._read_head.state_size == self._controller_network.output_spec
        #assert self._write_head.state_size == self._controller_network.output_spec
        # TODO: finish. Probably need to be more careful about input / output specs.

    def call(self, x_t, r_tm1, s_tm1, memory_tm1, read_args, write_args):
        s_t, _ = self._controller_network([x_t, r_tm1], s_tm1)
        y_t, _ = self._output_net(s_t)
        r_t, read_state = self._read_head([s_t, memory_tm1], *read_args)
        memory_t, write_state = self._write_head(s_t, memory_tm1, *write_args)
        return y_t, [r_t, s_t, memory_tm1, read_state, write_state]

    def __call__(self, x, *state):
        r_tm1 = state[0]
        s_tm1 = state[1]
        memory_tm1 = state[2]
        read_args = state[3]
        write_args = state[4]
        return self.call(x, r_tm1, s_tm1, memory_tm1, read_args, write_args)
