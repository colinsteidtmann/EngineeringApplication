import numpy as np
import sys
import random
import math
class NeuralNetwork:
    def __init__(self, sizes, activation="relu", scale_method="standardize", optimizer="adam"):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.

        ``activation`` is for the hidden layers, it must be a sting, either "sigmoid" or "tanh" or "relu" or "leaky_relu"
        ``scale_method`` is to scale the data to lies in a smaller range
        ``optimizer`` is to optimize the speed that our neural network learns. 
        """
        if not (activation == "sigmoid" or activation == "tanh" or activation == "relu" or activation == "leaky_relu"):
            sys.exit('Ooops! activation function must be "sigmoid" or "tanh" or "relu" or "leaky_relu"')
        if not (scale_method == "normalize" or scale_method == "standardize" or scale_method == "none"):
            sys.exit('Ooops! scale_method must be "normalize" or "standardize" or "none"')
        if not (optimizer == "sgd" or optimizer == "momentum" or optimizer == "nesterov" or optimizer == "adagrad" or optimizer == "rmsprop" or optimizer == "adam" or optimizer == "adamax" or optimizer == "nadam"):
            sys.exit('Ooops! optimizer must be "sgd," "momentum," "nesterov," "adagrad," "rmsprop," "adam," "adamax," or "nadam" ')
        elif (type(sizes) != list):
            sys.exit('Ooops! sized must be a list')
            
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.activation = activation
        self.scale_method = scale_method
        self.optimizer = optimizer
        self.initialize_weights()
        self.initialize_optimizer_params()
    
    def initialize_weights(self):
        """ Initlize our weights and biases with numbers drawn from a normal distribution 
            with mean=0 and std=1 
        """
        self.weights = [np.random.normal(0, (1/np.sqrt(inputSize)), (outputSize, inputSize)) for outputSize, inputSize in zip(self.sizes[1:], self.sizes[:-1])]
        self.biases = [np.random.normal(0, 1, (outputSize, 1)) for outputSize in self.sizes[1:]]
        self.copy_of_weights = np.copy(self.weights)
        self.copy_of_biases = np.copy(self.biases)

    def reset_weights(self):
        self.weights = self.copy_of_weights
        self.biases = self.copy_of_biases

    def initialize_optimizer_params(self):
        """ Initilize different optimizer paramaters. """

        if (self.optimizer == "momentum"):
            """ 
                "With Momentum update, the parameter vector will build up velocity in any direction that has consistent gradient."
                mu --> momentum rate, typical values are [0.5, 0.9, 0.95, 0.99]
                vws/vbs --> velocity and direction of gradients
            """
            self.mu = 0.9 
            self.vws = [np.zeros(w.shape) for w in self.weights]
            self.vbs = [np.zeros(b.shape) for b in self.biases]
        
        elif (self.optimizer == "nesterov"):
            """
                Nesterov momentum has the same paramaters as Momentum, 
                except it's implementation will be different
            """
            self.mu = 0.9 
            self.vws = [np.zeros(w.shape) for w in self.weights]
            self.vbs = [np.zeros(b.shape) for b in self.biases]
        
        elif (self.optimizer == "adagrad"):
            """
                Adagrad is an adaptive learning rate method, "weights that 
                receive high gradients will have their effective learning rate reduced, 
                while weights that receive small or infrequent updates will have their 
                effective learning rate increased."
                
                eps --> avoids division by zero, typical values range from 1e-4 to 1e-8
                sws/sbs --> keeps track of per-parameter sum of squared gradients
            """
            self.eps = 1e-8
            self.sws = [np.zeros(w.shape) for w in self.weights]
            self.sbs = [np.zeros(b.shape) for b in self.biases]
        
        elif (self.optimizer == "rmsprop"):
            """ 
                "The RMSProp update adjusts the Adagrad method in a very simple way in an 
                attempt to reduce its aggressive, monotonically decreasing learning rate. 
                In particular, it uses a moving average of squared gradients"

                gamma --> decay rate, typical values are [0.9, 0.99, 0.999]
            """
            self.gamma = 0.9
            self.eps = 1e-8
            self.sws = [np.zeros(w.shape) for w in self.weights]
            self.sbs = [np.zeros(b.shape) for b in self.biases]

        elif (self.optimizer == "adam" or self.optimizer == "adamax" or self.optimizer == "nadam"):
            """
                - Adam is RMSProp with momentum
                - Adamax is a stable version of Adam that is more robust to big gradients, 
                  it is better for when paramter are updated sparsly 
                - Nadam is Adam RMSprop with Nesterov momentum.
                mws/mbs --> "smooth" verion of the gradient instead of the raw (and perhaps noisy) 
                       gradient vector dx. 
            """
            self.eps = 1e-8
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.vws = [np.zeros(w.shape) for w in self.weights]
            self.vbs = [np.zeros(b.shape) for b in self.biases]
            self.mws = [np.zeros(w.shape) for w in self.weights]
            self.mbs = [np.zeros(b.shape) for b in self.biases]
        
    def SGD(self, training_data,lr=0.01, epochs=1):
        """ stochastic gradient descent - update weights at every step in batch 
             ``training_data`` is a list of tuples ``(x, y)`` representing the training inputs 
             and the desired outputs.
             ``epochs`` = number of iterations we want to go through the training data
             ``lr`` = learning rate, how fast our neural network learns the cost function gradient
        """
        
        self.total_costs = []
        if type(training_data) != list: training_data = list(training_data)
        if (self.scale_method != "none"): training_data = self.scale_fn(training_data)
        random.shuffle(training_data)
        print(training_data)
        for epoch in range(epochs):
            self.epoch_costs = []
            for x, y in training_data:
                nablaWs, nablaBs = self.backprop(x, y)
                self.optimize(nablaWs, nablaBs, lr, epoch)
            self.total_costs.append([epoch + 1, (sum(self.epoch_costs) / len(self.epoch_costs))[0][0]])
            #Stop early if cost is low enough
            if (self.total_costs[-1][1] < .0005):
                break
    def feedforward(self, inputs):
        """Return the outputs of the network if ``a`` is input
            Don't use activation function on last layer
        """
        if (self.scale_method != "none"): inputs = self.scale_fn(inputs)
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
    
    def optimize(self, nablaWs, nablaBs, lr, epoch):
        """ Updates paramaters with gradients of cost function w.r.t paramaters """
        if (self.optimizer == "sgd"):
            self.weights = [w - (lr * nw) for w,nw in zip(self.weights,nablaWs)]
            self.biases = [b - (lr * nb) for b, nb in zip(self.biases, nablaBs)]
            
        elif (self.optimizer == "momentum"):
            self.vws = [(self.mu * vw) + (lr * nw) for vw, nw in zip(self.vws, nablaWs)]
            self.vbs = [(self.mu * vb) + (lr * nb) for vb, nb in zip(self.vbs, nablaBs)]
            self.weights = [w - vw for w, vw in zip(self.weights, self.vws)]
            self.biases = [b - vb for b, vb in zip(self.biases, self.vbs)]

        elif (self.optimizer == "nesterov"):
            vws_prev = self.vws
            vbs_prev = self.vbs
            self.vws = [(self.mu * vw) + (lr * nw) for vw, nw in zip(self.vws, nablaWs)]
            self.vbs = [(self.mu * vb) + (lr * nb) for vb, nb in zip(self.vbs, nablaBs)]
            self.weights = [w - ((self.mu * vw_prev) + ((1 - self.mu) * vw)) for w, vw_prev, vw in zip(self.weights, vws_prev, self.vws)]
            self.biases = [b - ((self.mu * vb_prev) + ((1 - self.mu) * vb)) for b, vb_prev, vb in zip(self.biases, vbs_prev, self.vbs)]

        elif (self.optimizer == "adagrad"):
            self.sws = [sw + (nw ** 2) for sw, nw in zip(self.sws, nablaWs)]
            self.sbs = [sb + (nb ** 2) for sb, nb in zip(self.sbs, nablaBs)]
            self.weights = [w - (lr * nw / (np.sqrt(sw) + self.eps)) for w, sw, nw in zip(self.weights, self.sws, nablaWs)]
            self.biases = [b - (lr * nb / (np.sqrt(sb) + self.eps)) for b, sb, nb in zip(self.biases, self.sbs, nablaBs)]
         
        elif (self.optimizer == "rmsprop"):
            self.sws = [(self.gamma * sw) + ((1 - self.gamma) * (nw ** 2)) for sw, nw in zip(self.sws, nablaWs)]
            self.sbs = [(self.gamma * sb) + ((1 - self.gamma) * (nb ** 2)) for sb, nb in zip(self.sbs, nablaBs)]
            self.weights = [w - (lr * nw / (np.sqrt(sw) + self.eps)) for w, sw, nw in zip(self.weights, self.sws, nablaWs)]
            self.biases = [b - (lr * nb / (np.sqrt(sb) + self.eps)) for b, sb, nb in zip(self.biases, self.sbs, nablaBs)]
        
        elif (self.optimizer == "adam"):
            epoch = epoch + 1
            self.mws = [(self.beta1 * mw) + ((1 - self.beta1) * nw) for mw, nw in zip(self.mws, nablaWs)]
            self.mbs = [(self.beta1 * mb) + ((1 - self.beta1) * nb) for mb, nb in zip(self.mbs, nablaBs)]
            mtws = [mw / (1 - (self.beta1 ** epoch)) for mw in self.mws]
            mtbs = [mb / (1 - (self.beta1 ** epoch)) for mb in self.mbs]
            self.vws = [(self.beta2 * vw) + ((1 - self.beta2) * (nw ** 2)) for vw, nw in zip(self.vws, nablaWs)]
            self.vbs = [(self.beta2 * vb) + ((1 - self.beta2) * (nb ** 2)) for vb, nb in zip(self.vbs, nablaBs)]
            vtws = [vw / (1 - (self.beta2 ** epoch)) for vw in self.vws]
            vtbs = [vb / (1 - (self.beta2 ** epoch)) for vb in self.vbs]
            self.weights = [w - (lr * mtw / (np.sqrt(vtw) + self.eps)) for w, mtw, vtw in zip(self.weights, mtws, vtws)]
            self.biases = [b - (lr * mtb / (np.sqrt(vtb) + self.eps)) for b, mtb, vtb in zip(self.biases, mtbs, vtbs)]

        elif (self.optimizer == "adamax"):
            epoch = epoch + 1
            self.mws = [(self.beta1 * mw) + ((1 - self.beta1) * nw) for mw, nw in zip(self.mws, nablaWs)]
            self.mbs = [(self.beta1 * mb) + ((1 - self.beta1) * nb) for mb, nb in zip(self.mbs, nablaBs)]
            self.vws = [np.maximum((self.beta2 * vw), np.abs(nw)) for vw, nw in zip(self.vws, nablaWs)]
            self.vbs = [np.maximum((self.beta2 * vb), np.abs(nb)) for vb, nb in zip(self.vbs, nablaBs)]
            self.weights = [w - ((lr / (1 - (self.beta1 ** epoch))) * (mw/vw)) for w, mw, vw in zip(self.weights, self.mws, self.vws)]
            self.biases = [b - ((lr / (1 - (self.beta1 ** epoch))) * (mb/vb)) for b, mb, vb in zip(self.biases, self.mbs, self.vbs)]
        
        elif (self.optimizer == "nadam"):
            epoch = epoch + 1
            self.mws = [(self.beta1 * mw) + ((1 - self.beta1) * nw) for mw, nw in zip(self.mws, nablaWs)]
            self.mbs = [(self.beta1 * mb) + ((1 - self.beta1) * nb) for mb, nb in zip(self.mbs, nablaBs)]
            mtws = [((self.beta1 * mw) / (1 - (self.beta1 ** epoch))) + (((1 - self.beta1) * nw ) / (1 - (self.beta1 ** epoch))) for mw, nw in zip(self.mws, nablaWs)]
            mtbs = [((self.beta1 * mb) / (1 - (self.beta1 ** epoch))) + (((1 - self.beta1) * nb ) / (1 - (self.beta1 ** epoch))) for mb, nb in zip(self.mbs, nablaBs)]
            self.vws = [(self.beta2 * vw) + ((1 - self.beta2) * (nw ** 2)) for vw, nw in zip(self.vws, nablaWs)]
            self.vbs = [(self.beta2 * vb) + ((1 - self.beta2) * (nb ** 2)) for vb, nb in zip(self.vbs, nablaBs)]
            vtws = [(self.beta2 * vw) / (1 - (self.beta2 ** epoch)) for vw in self.vws]
            vtbs = [(self.beta2 * vb) / (1 - (self.beta2 ** epoch)) for vb in self.vbs]
            self.weights = [w - (lr * mtw / (np.sqrt(vtw) + self.eps)) for w, mtw, vtw in zip(self.weights, mtws, vtws)]
            self.biases = [b - (lr * mtb / (np.sqrt(vtb) + self.eps)) for b, mtb, vtb in zip(self.biases, mtbs, vtbs)]
            
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

    def scale_fn(self, x):
        if (self.scale_method == "standardize"):
            """ Standardizes data so μ = 0 and 𝛔 = 1 """
            if (type(x) == list):
                y = np.array(x)[:, 1]
                x = np.array(x)[:, 0]
                x = (x - x.min()) / (x.max() - x.min())
                return list(zip(x,y))
            return (x - x.min()) / (x.max() - x.min())
        elif (self.scale_method == "normalize"):
            if (type(x) == list):
                y = np.array(x)[:, 1]
                x = np.array(x)[:, 0]
                #print("before = ", x)
                x = (x - x.mean()) / (x.std())
                #print("after = ", x)
                return list(zip(x,y))
            return (x-x.mean())/(x.std())
        