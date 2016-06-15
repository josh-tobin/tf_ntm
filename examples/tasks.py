import numpy as np

def generate_copy_data(dimensionality, max_seq_length, num_examples):

    x = np.zeros([num_examples, 2 * max_seq_length + 2, dimensionality + 1])
    y = np.zeros([num_examples, 2 * max_seq_length + 2, dimensionality + 1])

    for example in range(num_examples):
        seq_length = np.random.randint(max_seq_length) + 1
        seq = np.random.randint(2, size=(seq_length, dimensionality))
        # Sequence
        x[example, :seq_length, :-1] = seq
        # Stop character
        x[example, seq_length, -1] = 1
        # Sequence
        y[example, (seq_length + 1):(2*seq_length + 1), :-1] = seq
        # Stop character
        y[example, (2*seq_length + 1), -1] = 1

    return x, y


