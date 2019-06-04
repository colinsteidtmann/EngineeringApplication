import numpy as np
import sys
class NeuralNetwork:
    def __init__(self, sizes, activation="relu", batchnorm=True):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.

        ``activation`` is for the hidden layers, it must be a sting, either "sigmoid" or "tanh" or "relu" or "leaky_relu"
        """
        if not (activation == "sigmoid" or activation == "tanh" or activation == "relu" or activation == "leaky_relu"):
            sys.exit('Ooops! activation function must be "sigmoid" or "tanh" or "relu" or "leaky_relu"')
        elif (type(sizes) != list):
            sys.exit('Ooops! sized must be a list')
            
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.activation = activation
        self.batchnorm = batchnorm
        self.initialize_weights()
        if self.batchnorm:self.initizlize_BN_params()
    
    def initialize_weights(self):
        """ Initlize our weights and biases with numbers drawn from a normal distribution 
            with mean=0 and std=1 
        """
        self.weights = [np.random.normal(0, 1, (outputSize, inputSize)) for outputSize, inputSize in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases = [np.random.normal(0, 1, (outputSize, 1)) for outputSize in self.sizes[1:]]
        
    def initizlize_BN_params(self):
        """ Initilize gammas (scaling) and betas (shifting) so they can learn to work better for 
            our chosen activation function
        """
        self.gammas = [np.ones((outputSize,1)) for outputSize in self.sizes[1:-1]]
        self.betas = [np.zeros((outputSize, 1)) for outputSize in self.sizes[1:-1]]
        
    def SGD(self, training_data, lr=0.01, epochs=1):
        """ stochastic gradient descent - update weights at every step in batch 
             ``training_data`` is a list of tuples ``(x, y)`` representing the training inputs 
             and the desired outputs.
             ``epochs`` = number of iterations we want to go through the training data
             ``lr`` = learning rate, how fast our neural network learns the cost function gradient
        """
        self.total_costs = []
        if type(training_data) != list: training_data = list(training_data)
        for epoch in range(epochs):
            self.epoch_costs = []
            for x, y in training_data:
                if self.batchnorm:
                    nablaWs, nablaBs, nablaGammas, nablaBetas = self.batchnorm_backprop(x, y)
                    self.weights = [w-(lr*nw) for w,nw in zip(self.weights,nablaWs)]
                    self.biases = [b - (lr * nb) for b, nb in zip(self.biases, nablaBs)]
                    self.gammas = [g - (lr * ng) for g, ng in zip(self.gammas, nablaGammas)]
                    self.betas = [b - (lr * nb) for b, nb in zip(self.betas, nablaBetas)]
                else:
                    nablaWs, nablaBs = self.backprop(x, y)
                    self.weights = [w-(lr*nw) for w,nw in zip(self.weights,nablaWs)]
                    self.biases = [b - (lr * nb) for b, nb in zip(self.biases, nablaBs)]
            self.total_costs.append([epoch+1, (sum(self.epoch_costs)/len(self.epoch_costs))[0][0]])
            
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
                    if self.batchnorm:
                        znorm, _, _ = self.znorm_fn(np.matmul(w, a) + b)
                        a = self.activation_fn(self.gammas[idx] * znorm + self.betas[idx])
                    else:
                        a = self.activation_fn(np.matmul(w, a) + b)
                    
            outputs.append(a)
        return outputs
    
    def backprop(self, x, y):
        """ reshape inputs and labels """
        x = np.array([x]).reshape((self.sizes[0], 1))
        y = np.array([y]).reshape((self.sizes[-1], 1))

        """ arrays to store weights and biases derivatives """
        nablaWs = [np.zeros(w.shape) for w in self.weights]
        nablaBs = [np.zeros(b.shape) for b in self.biases]

        """ arrays to store outputs of each layer """
        zLayers = [x]  
        aLayers = [x]
        
        """ feedforward with input x, store outputs, z, and activations of z of each layer """
        for w, b in zip(self.weights, self.biases):
            z = np.matmul(w, aLayers[-1]) + b
            zLayers.append(z)
            if (len(zLayers) == self.num_layers):
                a = z
            else:
                a = self.activation_fn(z)
            aLayers.append(a)
        
        """ Append cost of step to self.costs so we can plot our costs later """
        self.epoch_costs.append(self.cost_fn(aLayers[-1], y))

        """ Begin Backpropagation
            get d of cost w.r.t final layer,  δᴸ = ∇ₐC ⊙ σ′(zᴸ) 
            Multiply cost_prime * 1 because σ′(zᴸ) is linear so it equals 1
        """
        delta = self.cost_fn_prime(aLayers[-1], y) * 1
        
        """ backpropagate error to each layer in nn, store each d of cost w.r.t weight layer as nabla_w
             δˡ = ((wˡ⁺¹)ᵀδˡ⁺¹) ⊙ σ′(zˡ).
             ∇w = δˡ(aˡ-¹)ᵀ
             ∇b = δˡ
        """  
        for l in range(1, self.num_layers):
            nablaWs[-l] = np.dot(delta, aLayers[-l - 1].transpose())
            nablaBs[-l] = delta
            delta = np.dot(self.weights[-l].transpose(),delta) * self.activation_fn_prime(zLayers[-l-1])

        return nablaWs, nablaBs
    
    def batchnorm_backprop(self, x, y):
        """ reshape inputs and labels """
        x = np.array([x]).reshape((self.sizes[0], 1))
        y = np.array([y]).reshape((self.sizes[-1], 1))

        """ arrays to store weights and biases derivatives """
        nablaWs = [np.zeros(w.shape) for w in self.weights]
        nablaBs = [np.zeros(b.shape) for b in self.biases]
        nablaGammas = [np.zeros(g.shape) for g in self.gammas]
        nablaBetas = [np.zeros(b.shape) for b in self.betas]

        """ arrays to store outputs of each layer """
        zLayers = [x]
        meanLayers = []
        varLayers = []
        znormLayers = []
        aLayers = [x]
        
        """ feedforward with input x, store outputs, z, and activations of z of each layer """
        for w, b in zip(self.weights, self.biases):
            z = np.matmul(w, aLayers[-1]) + b
            zLayers.append(z)
            if (len(zLayers) == self.num_layers):
                a = z
            else:
                l = len(znormLayers)
                znorm,mean,var = self.znorm_fn(z)
                znormLayers.append(znorm)
                meanLayers.append(mean)
                varLayers.append(var)
                a = self.activation_fn((self.gammas[l]*znorm+self.betas[l]))
            aLayers.append(a)
        
        """ Append cost of step to self.costs so we can plot our costs later """
        self.epoch_costs.append(self.cost_fn(aLayers[-1], y))

        """ Begin Backpropagation
            get d of cost w.r.t final layer,  δᴸ = ∇ₐC ⊙ σ′(zᴸ) 
            Multiply cost_prime * 1 because σ′(zᴸ) is linear so it equals 1
        """
        delta = self.cost_fn_prime(aLayers[-1], y) * 1
        
        """ backpropagate error to each layer in nn, store each d of cost w.r.t weight layer as nabla_w
             δˡ = ((wˡ⁺¹)ᵀδˡ⁺¹) ⊙ σ′(zˡ).
             ∇w = δˡ(aˡ-¹)ᵀ
             ∇b = δˡ
        """  
        for l in range(1, self.num_layers):
            nablaWs[-l] = np.dot(delta, aLayers[-l - 1].transpose())
            nablaBs[-l] = delta
            if l <= len(nablaGammas):
                dout = np.dot(self.weights[-l].transpose(),delta) * self.activation_fn_prime(znormLayers[-l])
                nablaGammas[-l] = dout * znormLayers[-l]
                nablaBetas[-l] = dout
                delta = (1/len(dout))*(1/np.sqrt(varLayers[-l]+1e-6))*(len(dout)*(dout*self.gammas[-l])-np.sum(dout*self.gammas[-l])-znormLayers[-l]*np.sum((dout*self.gammas[-l])*znormLayers[-l]))
            else:
                delta = np.dot(self.weights[-l].transpose(),delta) * self.activation_fn_prime(zLayers[-l-1])

        return nablaWs, nablaBs, nablaGammas, nablaBetas
    
    def activation_fn(self, z):
        if self.activation == "sigmoid":
            return 1.0 / (1.0+ np.exp(-z))
        elif self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "leaky_relu":
            return np.maximum(0.01 * z, z)
    
    def activation_fn_prime(self, z):
        if self.activation == "sigmoid":
            return self.activation_fn(z) * (1 - self.activation_fn(z))
        elif self.activation == "tanh":
            return (1 - (np.tanh(z)** 2))
        elif self.activation == "relu":
            z[z <= 0] = 0
            z[z > 0] = 1
            return z
        elif self.activation == "leaky_relu":
            z[z <= 0] = 0.01
            z[z > 0] = 1
            return z
    
    def cost_fn(self, a, y):
        return 0.5 * ((y - a)** 2)
    
    def cost_fn_prime(self, a, y):
        return (a - y)

    def znorm_fn(self, z):
        mean = np.mean(z)
        var = np.var(z)
        sd = np.sqrt(var + 1e-6)
        znorm = (z - mean) / sd
        return znorm,mean,var

    