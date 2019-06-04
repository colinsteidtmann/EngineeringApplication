import numpy as np
import matplotlib.pyplot as plt
import neuralnetwork_ex7 as neuralnetwork
nn = neuralnetwork.NeuralNetwork(sizes=[1,256,1], activation="relu", scale_method="standardize", optimizer="nadam") 
plt.ion()
fig, ax = plt.subplots()

plt.figure(1)
""" Generate x's and sine(x's) """
x = np.arange(0, np.pi * 8, 0.1)
y = np.sin(x)
ax.plot(x, y)
ax.scatter(x, y)

predictions = np.zeros(len(x))
sc = ax.scatter(x, predictions)
pc, = ax.plot(x,predictions)
plt.xlim(-5,45)
plt.ylim(-2,4)

plt.draw()
for training in range(5000):
    predictions = np.squeeze(nn.feedforward(x))
    nn.SGD(zip(x, y), lr=.001, epochs=1, num=training)
    print(nn.total_costs[0][1], training)
    if nn.total_costs[0][1] < .0005:
        break
    sc.set_offsets(np.c_[x, predictions])
    pc.set_ydata(predictions)
    fig.canvas.draw_idle()
    plt.pause(0.1)
    

""" Plot our after training predictions """
predictions = np.squeeze(nn.feedforward(x))
plt.figure(2)
plt.plot(x,predictions)

plt.show()