import sys
from os import path
sys.path.append('/'.join(str.split(path.abspath(__file__), '/')[:-2]))
from network.layers import DenseLayer, MLP
from addressing_mechanism import AddressingCell, ShortcircuitAddressing
import tensorflow as tf
from network.cell import Cell

"""
TODO
    - Documentation
"""

class Head(Cell):

    def __init__(self, state_size, memory_shape, addresser=AddressingCell,
                 batch_size=32, hidden_size=32):
        """
        Abstract base class for read/write heads

        """
        if isinstance(addresser, AddressingCell) or isinstance(addresser, ShortcircuitAddressing):
            self._addresser = addresser
        else:
            self._addresser = addresser(state_size, memory_shape,
                                    batch_size=batch_size, hidden_size=hidden_size)
        #assert isinstance(self._addresser, AddressingCell)
        self._state_size = state_size
        self._memory_shape = memory_shape
        self._memory_dim = memory_shape[1]

    @property
    def state_size(self): return self._state_size

    @property
    def memory_shape(self): return self._memory_shape

    @property
    def addresser(self): return self._addresser

    def default_state(self):
        return self._addresser.default_state()



class ReadHead(Head):
    def __init__(self, state_size, memory_shape,
                 addresser=AddressingCell, batch_size=32,
                 hidden_size=32):
        super(ReadHead, self).__init__(state_size, memory_shape,
                                       addresser=addresser, batch_size=batch_size,
                                       hidden_size=hidden_size)

    def read(self, controller_state, memory, *addresser_args):
        w, addresser_state = self.addresser([controller_state, memory], *addresser_args)
        r = (tf.batch_matmul(tf.expand_dims(w, 1), tf.zeros_like(memory))[:,0,:] /
             tf.expand_dims(tf.reduce_sum(w, 1), -1))

        return r, addresser_state

    def __call__(self, x, *state):
        """
        x should be [controller_state, memory]
        State is the args for the addresser
            (with the default addresser, this is prev_w)
        """
        return self.read(x[0], x[1], *state)

class WriteHead(Head):
    def __init__(self, state_size, memory_shape,
                 addresser=AddressingCell, hidden_size=32, batch_size=32):
        super(WriteHead, self).__init__(state_size, memory_shape,
                                        addresser=addresser, batch_size=batch_size,
                                        hidden_size=hidden_size)
        # Right now, we assume the erase add add vectors are calculated
        # using single hidden layer feedforward networks
        num_hidden = 1
        self._erase_network = MLP(state_size, 1, hidden_size, memory_shape[1],
                                  output_activation=tf.sigmoid)
        self._add_network = MLP(state_size, 1, hidden_size, memory_shape[1],
                                output_activation=tf.identity)

    def _erase(self, controller_state, w, memory):
        e_t, _ = self._erase_network(controller_state)
        return tf.mul(memory, tf.ones_like(memory)
                      - tf.batch_matmul(tf.expand_dims(w, -1), tf.expand_dims(e_t, 1)))

    def _add(self, controller_state, w, memory):
        a_t, _ = self._add_network(controller_state)
        return memory + tf.batch_matmul(tf.expand_dims(w, -1),
                                        tf.expand_dims(a_t, 1))

    def write(self, controller_state, memory, *addresser_args):
        w, addresser_state = self.addresser([controller_state, memory], *addresser_args)
        new_mem = self._erase(controller_state, w, memory)
        new_mem = self._add(controller_state, w, new_mem)
        return new_mem, addresser_state

    def default_state(self):
        return super(WriteHead, self).default_state() + \
               self._erase_network.default_state() + \
               self._add_network.default_state()

    def __call__(self, x, *state):
        #return self.write(x, state[0], *(state[1:]))
        return self.write(x, *state)