import sys
from os import path
sys.path.append('/'.join(str.split(path.abspath(__file__), '/')[:-2]))
from network.layers import MLP, DenseLayer
from network.network import Network
from util.activations import relu
from ntm.head import Head
from ntm.ntm_cell import NTMCell
from tasks import generate_copy_data
import tensorflow as tf
import argparse

def copy_network(input_dimension=8, batch_size=32, state_size=16,
                 max_seq_length=8, memory_size=128,
                 controller_layers=1, controller_layer_size=32,
                 head_hidden_size=8):

    controller_network = MLP((input_dimension + 1, input_dimension),
                             controller_layers, controller_layer_size, state_size)

    x = tf.placeholder(tf.float32, [batch_size, max_seq_length*2 + 2,
                                    input_dimension + 1])
    # Note: even though we want sigmoids, the loss function expects
    # unscaled logits so we do not apply sigmoids here.
    output_cell = DenseLayer(state_size, input_dimension + 1,
                             activation=tf.identity)
    ntm_cell = NTMCell(controller_network,
                       (memory_size, input_dimension), batch_size,
                       head_hidden_size=head_hidden_size,
                       output_net=output_cell)
    ntm = Network(ntm_cell, x)
    return ntm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_training_examples', type=int, default=3200)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_result', type=bool, default=False)
    parser.add_argument('--save_location', default='default')

    parser.add_argument('--num_test_examples', type=int, default=20)
    parser.add_argument('--input_dimension', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--state_size', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=5)
    parser.add_argument('--memory_size', type=int, default=64)
    parser.add_argument('--controller_layers', type=int, default=1)
    parser.add_argument('--controller_layer_size', type=int, default=32)
    parser.add_argument('--head_hidden_size', type=int, default=8)
    args = parser.parse_args()

    print 'Building Neural Turing Machine...'
    ntm = copy_network(
        input_dimension=args.input_dimension,
        batch_size=args.batch_size,
        state_size=args.state_size,
        max_seq_length=args.max_seq_length,
        memory_size=args.memory_size,
        controller_layers=args.controller_layers,
        controller_layer_size=args.controller_layer_size,
        head_hidden_size=args.head_hidden_size
    )

    #optimizer = tf.train.AdamOptimizer()
    optimizer = tf.train.GradientDescentOptimizer(1e-4)
    loss = tf.nn.sigmoid_cross_entropy_with_logits
    print 'Compiling Neural Turing Machine...'
    ntm.compile(loss, optimizer)

    print
    print 'Generating training data...'
    x, y = generate_copy_data(args.input_dimension, args.max_seq_length,
                       args.num_training_examples)

    print
    print 'Training!'
    ntm.train(x, y, batch_size=args.batch_size, n_epochs=args.num_epochs)


if __name__ == '__main__':
    main()