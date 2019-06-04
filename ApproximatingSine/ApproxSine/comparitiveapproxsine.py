import numpy as np
import matplotlib.pyplot as plt
import neuralnetwork_ex7 as neuralnetwork
import math

""" Create a feed-forward neural network using our neuralnetwork_ex6.py file """
NN = neuralnetwork.NeuralNetwork(sizes=[1,128,1], activation="relu", scale_method="normalize", optimizer="sgd") 

""" Generate x's and sine(x's) """
x = np.arange(0, np.pi * 8, 0.1)
y = np.sin(x)

""" Matplotlib, plot the  sine wave"""
plt.figure(1)
plt.plot(x, y)

""" Array to store the optimizers we will run through """
# optimizersArray = ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adam", "adamax", "nadam"]
optimizersArray = ["sgd","nesterov","nadam"]
lrArray = [1e-4,1e-3,1e-2,1e-1]

for optimizer in optimizersArray:
        """ Update optimizer of neural network """
        NN.optimizer = optimizer
        NN.initialize_optimizer_params()
        for lr in lrArray:
                NN.reset_weights()
                """ Perform backpropagation 1000 times, keeping track of the cost for each iteration through training data """
                iterations = []
                costs = []
                NN.SGD(zip(x, y), lr=lr, epochs=5000)
                for iteration, cost in NN.total_costs:
                        iterations.append(iteration)
                        costs.append(cost)


                """ Plot our after training predictions """
                predictions = np.squeeze(NN.feedforward(x))
                plt.figure(1)
                plt.plot(x,predictions, label="Optimizer: {} (lr={})".format(optimizer,lr))

                """ Plot the costs for each iteration """
                plt.figure(2)
                plt.plot(iterations,costs, label="Optimizer: {} (lr={})".format(optimizer,lr))
                print("optimizer: ", optimizer, "lr: ", lr, " final cost = ", costs[-1])

plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()
plt.show()