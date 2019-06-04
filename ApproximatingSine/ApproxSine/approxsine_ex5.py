import numpy as np
import matplotlib.pyplot as plt
import neuralnetwork_ex5 as neuralnetwork

""" Create a feed-forward neural network using our neuralnetwork_ex5.py file """
NN = neuralnetwork.NeuralNetwork(sizes=[1,512,1], activation="relu", scale_method="normalize", optimizer="nadam") 

""" Generate x's and sine(x's) """
x = np.arange(0, np.pi * 16, 0.1)
y = np.sin(x)

""" Matplotlib, plot the  sine wave"""
plt.figure(1)
plt.plot(x, y)

""" Plot our before training predictions """ 
predictions = np.squeeze(NN.feedforward(x))
plt.figure(1)
plt.plot(x, predictions)

""" Perform backpropagation 1000 times, keeping track of the cost for each iteration through training data """
iterations = []
costs = []
NN.SGD(zip(x, y), lr=0.001, epochs=1000)
for iteration, cost in NN.total_costs:
        iterations.append(iteration)
        costs.append(cost)

""" Plot our after training predictions """
predictions = np.squeeze(NN.feedforward(x))
plt.figure(1)
plt.plot(x,predictions)

""" Plot the costs for each iteration """
plt.figure(2)
plt.plot(iterations,costs)
print("final cost = ", costs[-1])

plt.show()