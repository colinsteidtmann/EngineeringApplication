import random
import numpy as np
import sys


class Network:
    def __init__(self, sizes, cost, activation="leaky_relu", optimizer="adam", dropout=True):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.

        ``cost`` must be a sting, either "quadratic" or "crossEntropy"
        ``activation`` is for the hidden layers, it must be a sting, either "leaky_relu" or "relu" or "sigmoid"
        ``optimizer`` must be a string, either "adam" or "sgd" (standard gradient descent)
        ``dropout`` is a boolean which reffers to dropping neurons in hidden layers to keep our NN from overfitting
        """

        if not (cost == "quadratic" or cost == "cross_entropy"):
            sys.exit('Ooops! cost function must be "quadratic" or "cross_entropy"')
        elif not (activation == "leaky_relu" or activation == "relu" or activation == "sigmoid" or activation == "tanh"):
            sys.exit('Ooops! activation function must be "leaky_relu" or "relu" or "sigmoid" or "tanh"')
        elif not (optimizer == "adam" or optimizer == "sgd"):
            sys.exit('Ooops! optimizer must be "adam" or "sgd"')
        elif (type(sizes) != list):
            sys.exit('Ooops! sized must be a list')
            
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.cost = cost
        self.activation = activation
        self.optimizer = optimizer
        self.initialize_weights()

        """ variables initilized if optimizer is "adam" """
        self.vdws = [0 for w in range(len(self.weights))]
        self.sdws = [0 for w in range(len(self.weights))]
        self.vdbs = [0 for w in range(len(self.biases))]
        self.sdbs = [0 for w in range(len(self.biases))]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08

        """Only applies if using dropout"""
        self.dropout = dropout
        self.dropout_input_probability = 1.0
        self.dropout_hl_probability = 0.5

    
    def initialize_weights(self):
        # self.weights = [np.random.rand(outputSize, inputSize) for outputSize, inputSize in zip(self.sizes[1:], self.sizes[:-1])]
        # self.biases = [np.random.rand(outputSize, 1) for outputSize in self.sizes[1:]]
        # self.weights = [np.zeros((outputSize, inputSize)) for outputSize, inputSize in zip(self.sizes[1:], self.sizes[:-1])]
        # self.biases = [np.zeros((outputSize, 1)) for outputSize in self.sizes[1:]]
        self.weights = [np.random.normal(0, (1/np.sqrt(inputSize)), (outputSize, inputSize)) for outputSize, inputSize in zip(self.sizes[1:], self.sizes[:-1])]
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
    

    def SGD(self, training_data, lr=0.01, mini_batch_size=8, epochs=1, updateLoss=False):
        """ stochastic gradient descent - update weights at every step in batch 
             ``training_data`` is a list of tuples ``(x, y)`` representing the training inputs 
             and the desired outputs.
        """
        if type(training_data) != list: training_data = list(training_data)
        if updateLoss: self.lossArray = []
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batch = training_data[:mini_batch_size]
            for x, y in mini_batch:
                nablaWs, nablaBs = self.backprop(x, y, addLoss=updateLoss)
                self.optimize(nablaWs, nablaBs, lr, epoch)
                # self.weights = [w-(lr*nw) for w,nw in zip(self.weights,nablaWs)]
                # self.biases = [b-(lr*nb) for b,nb in zip(self.biases,nablaBs)]
        if updateLoss: self.episodeLoss = (np.sum(self.lossArray)/len(self.lossArray))
    
    def BGD(self, training_data, lr=0.01, mini_batch_size=8, epochs=1, updateLoss=False):
        """ batch gradient descent - update weights at end of batch, averagting sum of step updates """
        if type(training_data) != list: training_data = list(training_data)
        if updateLoss: self.lossArray = []
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batch = training_data[:mini_batch_size]
            sumNablaWs = [np.zeros(w.shape) for w in self.weights]
            sumNablaBs = [np.zeros(b.shape) for b in self.biases]
            for x, y in mini_batch:
                nablaWs, nablaBs = self.backprop(x, y, addLoss=updateLoss)
                sumNablaWs = [snw + nw for snw, nw in zip(sumNablaWs, nablaWs)]
                sumNablaBs = [snb + nb for snb, nb in zip(sumNablaBs, nablaBs)]
            self.optimize(sumNablaWs, sumNablaBs, (lr/mini_batch_size), epoch)
            # self.weights = [w-((lr/mini_batch_size)*snw) for w,snw in zip(self.weights,sumNablaWs)]
            # self.biases = [b-((lr/mini_batch_size)*snb) for b,snb in zip(self.biases,sumNablaBs)]
        if updateLoss: self.episodeLoss = (np.sum(self.lossArray)/len(self.lossArray))
    
    def backprop(self, x, y, addLoss=False):
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
            """ Drop neurons if randomly if using dropout """
            if (self.dropout == True):
                aLayers[-1] = self.applyDropout(aLayers[-1], len(aLayers))
                
            z = np.matmul(w, aLayers[-1]) + b
            zLayers.append(z)
            if (len(zLayers) == self.num_layers):
                a = z
            else:
                a = self.activation_fn(z)
            aLayers.append(a)
        
        """ Add loss to losses array if wanted """
        if (addLoss):
            self.lossArray.append(self.cost_fn(aLayers[-1], y))

        """ Begin Backpropagation
            get d of cost w.r.t final layer,  δᴸ = ∇ₐC ⊙ σ′(zᴸ) 
            Multiply cost_prime * 1 because σ′(zᴸ) is linear so it equals 1
        """
        delta = self.cost_prime(aLayers[-1], y) * 1
        
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
        """ If optimizer is gradient_descent """
        if (self.optimizer == "gradient_descent"):
            self.weights = [w-(lr*nw) for w,nw in zip(self.weights,nablaWs)]
            self.biases = [b-(lr*nb) for b,nb in zip(self.biases,nablaBs)]
        
        """ If optimizer is adam """
        if (self.optimizer == "adam"):
            epoch = epoch + 1
            """Update Weights"""
            self.vdws = [(self.beta1*vdw + (1-self.beta1)*nw) for  vdw,nw in zip(self.vdws,nablaWs)]
            self.sdws = [(self.beta2*sdw + (1-self.beta2)*(nw**2)) for sdw, nw in zip(self.sdws, nablaWs)]
            vdws_corrected = [(vdw/(1-self.beta1**epoch)) for vdw in self.vdws]
            sdws_corrected = [(sdw/(1-self.beta2**epoch)) for sdw in self.sdws]
            self.weights = [(w-lr*(vdw_corrected/(np.sqrt(sdw_corrected)+self.epsilon))) for w, vdw_corrected, sdw_corrected, in zip(self.weights, vdws_corrected, sdws_corrected)]
            """Update Biases"""
            self.vdbs = [(self.beta1*vdb + (1-self.beta1)*nb) for  vdb,nb in zip(self.vdbs,nablaBs)]
            self.sdbs = [(self.beta2*sdb + (1-self.beta2)*(nb**2)) for sdb, nb in zip(self.sdbs, nablaBs)]
            vdbs_corrected = [(vdb/(1-self.beta1**epoch)) for vdb in self.vdbs]
            sdbs_corrected = [(sdb / (1 - self.beta2 ** epoch)) for sdb in self.sdbs]
            self.biases = [(b-lr*(vdb_corrected/(np.sqrt(sdb_corrected)+self.epsilon))) for b, vdb_corrected, sdb_corrected, in zip(self.biases, vdbs_corrected, sdbs_corrected)]

    def cost_fn(self, a, y):
        if (self.cost == "quadratic"):
            return 0.5*((y-a)**2)
        elif (self.cost == "cross_entropy"):
            return (np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))))

    def cost_prime(self, a, y):
        if (self.cost == "quadratic"):
            return (a - y)
        elif (self.cost == "cross_entropy"):
            return ((a - y) / ((1 - a) * a))


    def activation_fn(self, z):
        if self.activation == "leaky_relu":
            return np.maximum(0.01 * z, z)
        elif self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0+ np.exp(-z))
        elif self.activation == "tanh":
            return np.tanh(z)


    def activation_fn_prime(self, z):
        if self.activation == "leaky_relu":
            z[z <= 0] = 0.01
            z[z > 0] = 1
            return z
        elif self.activation == "relu":
            z[z <= 0] = 0
            z[z > 0] = 1
            return z
        elif self.activation == "sigmoid":
            return self.activation_fn(z) * (1 - self.activation_fn(z))
        elif self.activation == "tanh":
            return (1 - (np.tanh(z)** 2))

    def applyDropout(self, a, layerNum):
        if (layerNum == 1):
            probs = np.random.binomial(size=a.shape, n=1, p=self.dropout_input_probability)
        else:
            probs = np.random.binomial(size=a.shape, n=1, p=self.dropout_hl_probability)
        return (a*probs)

            