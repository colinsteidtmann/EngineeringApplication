import numpy as np
import matplotlib.pyplot as plt
import neuralnetwork_ex8 as neuralnetwork

""" Create a feed-forward neural network using our neuralnetwork_ex8.py file """
NN = neuralnetwork.NeuralNetwork(sizes=[1, 512, 1], activations=["linear", "relu", "linear"], scale_method="normalize", optimizer="nadam", lr=0.001)
NN.set_regularizer(regularizer="early_stopping", patience=100)

""" Generate x's and sine(x's) """
x = np.arange(0, np.pi * 4, 0.1)
y = np.sin(x)

""" Matplotlib, plot the  sine wave"""
plt.figure(1)
plt.plot(x, y)

""" Plot our before training predictions """ 
predictions = np.squeeze(NN.feedforward(x))
plt.figure(1)
plt.plot(x, predictions)

""" Perform backpropagation 1000 times, keeping track of the cost for each iteration through training data """
train_iterations = []
train_costs = []
test_iterations = []
test_costs = []
NN.SGD_Fit(zip(x, y), epochs=100000, train_pct=0.8)
# gradients = [NN.gradients(x, y, 2, 3, "weights", "loss") for x, y in zip(x, y)]
for sgd_train_costs, sgd_test_costs in zip(NN.sgd_train_costs, NN.sgd_test_costs):
        train_iterations.append(sgd_train_costs[0])
        train_costs.append(sgd_train_costs[1])
        test_iterations.append(sgd_test_costs[0])
        test_costs.append(sgd_test_costs[1])

""" Plot our after training predictions """
predictions = np.squeeze(NN.feedforward(x))
plt.figure(1)
plt.plot(x,predictions)

""" Plot the costs for each iteration """
plt.figure(2)
plt.plot(train_iterations, train_costs, label="training cost")
plt.plot(test_iterations, test_costs, label="testing cost")
plt.legend()

plt.show()
