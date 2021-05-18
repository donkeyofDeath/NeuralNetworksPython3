import sys
import Network as nw
import numpy as np

sys.path.append("../")

import mnist_loader

# Load the data for the neural network.
train_data, validation_data, verification_data = mnist_loader.load_data_wrapper()

# Parameters for the neural network.
layer_sizes = [784, 30, 10]
weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

net = nw.Network(layer_sizes, weights, biases)  # Declare the network.

net.learn(train_data, 30, 10, 3.0, test_data=verification_data)  # Let the network learn.
