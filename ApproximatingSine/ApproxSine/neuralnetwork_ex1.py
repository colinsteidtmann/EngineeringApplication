import numpy as np
import sys
class NeuralNetwork:
    def __init__(self, sizes, activation="relu"):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.

        ``activation`` is for the hidden layers, it must be a sting, either "sigmoid" or "tanh" or "relu"
        """
        if not (activation == "sigmoid" or activation == "tanh" or activation == "relu"):
            sys.exit('Ooops! activation function must be "sigmoid" or "tanh" or "relu"')
        elif (type(sizes) != list):
            sys.exit('Ooops! sized must be a list')
            
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.activation = activation
        self.initialize_weights()
    
    def initialize_weights(self):
        """ Initlize our weights and biases with numbers drawn from a normal distribution 
            with mean=0 and std=1 
        """
        self.weights = [np.random.normal(0, 1, (outputSize, inputSize)) for outputSize, inputSize in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases = [np.random.normal(0, 1, (outputSize, 1)) for outputSize in self.sizes[1:]]
    
    def feedforward(self, inputs):
        """Return the outputs of the network if ``a`` is input
            Don't use activation function on last layer
        """
        outputs = []
        for a in inputs:
            a = np.array([a]).reshape((self.sizes[0],1))
            for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
                if idx == (len(self.weights) - 1):
                    a = np.matmul(w, a) + b
                else:
                    a = self.activation_fn(np.matmul(w, a) + b)
            outputs.append(a)
        return outputs
    
    def activation_fn(self, z):
        if self.activation == "sigmoid":
            return 1.0 / (1.0+ np.exp(-z))
        elif self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "relu":
            return np.maximum(0, z)