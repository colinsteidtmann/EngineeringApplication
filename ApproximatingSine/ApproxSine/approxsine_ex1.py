import numpy as np
import matplotlib.pyplot as plt
import neuralnetwork_ex1 as neuralnetwork

""" Create a feed-forward neural network using our neuralnetwork_ex1.py file """
NN = neuralnetwork.NeuralNetwork(sizes=[1,4,1],activation="sigmoid") 

""" Generate x's and sine(x's) """
x = np.arange(0, np.pi * 2 , 0.1)
y = np.sin(x)

""" Matplotlib, plot the  sine wave"""
plt.plot(x,y)


""" Use our feed-forward neural network to make predictions for each x """
predictions = np.squeeze(NN.feedforward(x))

""" plot our predictions """
plt.plot(x,predictions)

plt.show()