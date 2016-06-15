"""
Contains the Cell class. Cells are the building blocks of (recurrent)
networks.
"""

class ConnectorSpecification(object):
    def __init__(self, list_of_sizes):
        """
        Defines the specification of the input / output of a Cell. Used
        to test whether cells are compatible to be stacked.

        :param list_of_sizes: a list of tuples, where each tuple corresponds
                              to the specification for that input / output
        """
        if isinstance(list_of_sizes, tuple):
            list_of_sizes = [list_of_sizes]
        for size in list_of_sizes:
            assert isinstance(size, tuple)
        self._spec = list_of_sizes

    def is_empty(self):
        return self._spec == [()]

    def compatible(self, other):
        """
        Test whether two specifications are compatible, defined as
        whether the corresponding tensors can be substituted for
        one another

        :param other: another ConnectorSpecification object
        :return: boolean
        """
        compatible = True
        # If either specification is empty, then they are vacuously compatible
        if self.is_empty() or other.is_empty():
            return compatible
        # If they have different numbers of tensors, we cannot subsitute
        if len(self._spec) != len(other._spec):
            compatible = False
            return compatible
        else:
            # Check if each of the individual tensors are compatible
            for s, o in zip(self._spec, other._spec):
                # If they are different lengths, incompatible
                if len(s) != len(o):
                    compatible = False
                    return compatible
                # Otherwise, look at each dimension
                else:
                    for s_i, o_i in zip(s, o):
                        if s_i is not None and o_i is not None and s_i != o_i:
                            compatible = False
                            return compatible
        return compatible

    def __eq__(self, other):
        return self._spec == other._spec


class Cell(object):
    def __init__(self):
        """
        Abstract base class for Cells, the building block of neural network
        models. Cells can be thought of as a single timestep of a recurrent model.

        They map (1) x, a (list of) 2-d tensor(s) of shape (batch_size, input_dim)
        at this timestep, and (2) the state from the previous timestep
        to an output and state for the current timestep.

        Cells are callable, and batched by default. If recurrent connections are
        present, we will usually use them by making them part of Networks,
        which consider the operation of the cell rolled out over time.
        """
        self._input_spec = ConnectorSpecification(())
        self._output_spec = ConnectorSpecification(())

    def __call__(self, x, *state):
        raise NotImplementedError("Abstract method")

    @property
    def input_spec(self): return self._input_spec

    @property
    def output_spec(self): return self._output_spec

    def default_state(self):
        """
        Default initial state for the cell
        """
        return []