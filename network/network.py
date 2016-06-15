import tensorflow as tf
import numpy as np
from utils import slice_rnn_input, combine_rnn_output, shuffle_data

# TODO: refactor as a Network class and separate (supervised) model class

class Network(object):
    def __init__(self, cell, input_, initial_state=None):
        """
        Network objects represent the action of a cell on a particular input.
        In particular, for cells with recurrent connections, they roll out
        the recurrent connection in the time dimension.
        :param cell: a Cell object defining the action at each timestep
        :param input: a tensor (or list of tensors) representing the inputs
                      to the network
        :param initial_state: a list of tensors representing the recurrent
                              state at t=0
        """
        self._cell = cell
        self._input = input_
        self._reformated_input = self._reformat_input(input_)
        if initial_state is None:
            self._initial_state = self._cell.default_state()
        else:
            self._initial_state = initial_state

        self._compiled = False

    @property
    def input(self): return self._input

    @property
    def cell(self): return self._cell

    @property
    def initial_state(self): return self._initial_state

    @property
    def y(self):
        if self._compiled:
            return self._y
        else:
            raise Exception("Must compile network first!")

    def loss(self):
        if self._compiled:
            return self._loss
        else:
            raise Exception("Must compile network first!")

    def train_step(self):
        if self._compiled:
            return self._train_step
        else:
            raise Exception("Must compile network first!")

    def output(self):
        state = self._initial_state
        outputs = []
        for input_ in self._reformated_input:
            output, state = self._cell(input_, *state)
            outputs.append(output)
        output = self._reformat_output(outputs)
        return output, state

    def compile(self, loss_function, optimizer):
        self._optimizer = optimizer
        y_pred = self.output()[0]
        if isinstance(y_pred, list): y_pred = y_pred[0]
        self._y = tf.placeholder(tf.float32, y_pred.get_shape())
        self._loss = loss_function(self._y, y_pred)
        self._train_step = self._optimizer.minimize(self._loss)


        self._sess = tf.Session()
        self._sess.run(tf.initialize_all_variables())
        self._compiled = True

    def train(self, x, y, batch_size=32, n_epochs=100, shuffle=True, verbose=True,
              print_interval=1):
        losses = []
        n_batches = x.shape[0] // batch_size
        for epoch in range(n_epochs):
            x, y = shuffle_data(x, y)
            x_batches = np.split(x, n_batches)
            y_batches = np.split(y, n_batches)
            epoch_losses = []
            for x_batch, y_batch in zip(x_batches, y_batches):
                dict = {self.input: x_batch, self.y: y_batch}
                self._sess.run(self._train_step, feed_dict=dict)
                loss = self._sess.run(self._loss, feed_dict=dict)
                epoch_losses.append(loss)
            if verbose and (epoch % print_interval) == 0:
                print "Epoch %d/%d, loss = %f."%(epoch, n_epochs,
                                                 float(np.mean(epoch_losses)))
            losses.append(np.mean(epoch_losses))
        return losses

    def evaluate(self, x):
        if not self._compiled:
            raise Exception("Must compile network first!")
        return self._sess.run(self.output()[0], feed_dict={self.input: x})

    def test(self, x, y):
        if not self._compiled:
            raise Exception("Must compile network first!")
        return self._sess.run(self._loss, feed_dict={self.input: x,
                                                     self.y: y})


    def _reformat_output(self, outputs):
        # If we were originally passed a tensor input (vs. a list),
        # return a tensor
        if not self._return_list:
            #if len(outputs) == 1:
            #    return outputs[0]
            #else:
            return combine_rnn_output(outputs)

        else:
            return outputs

    def _reformat_input(self, input_):
        if not isinstance(input_, list):
            # If we are fed a tensor instead of a list, return a tensor
            self._return_list = False
            return slice_rnn_input(input_)
        else:
            return input_

