import matplotlib.pyplot as plt
import numpy as np
import random

inputs = np.random.rand(1,40)*10
outputs = np.sin(inputs)
plt.scatter(inputs, outputs)

weights1 = np.random.rand(1, 16)
bias1 = np.random.rand(1,16)
weights2 = np.random.rand(16, 1)
bias2 = np.random.rand(1,1)

for _ in range(50):
    xs = []
    ys = []
    for i in range(len(inputs[0])):
        hl = np.dot(inputs[0][i], weights1) + bias1
        hl[hl <= 0] = 0
        output = np.dot(hl, weights2) + bias2
        hl[hl <= 0] = 0
        hl[hl > 0] = 1
        errorGradient = (outputs[0][i]-output)
        weights2Gradient = np.dot(np.dot(hl.reshape(16,1),errorGradient), np.array(inputs[0][i]).reshape((1, 1)))
        weights1Gradient = np.dot(errorGradient, hl)
        bias2Gradient = np.dot(errorGradient, bias2)
        bias1Gradient = np.dot(errorGradient, bias1)
        weights2 = (0.99 * weights2) + (.01 * weights2Gradient)
        weights1 = (0.99 * weights1) + (.01 * weights1Gradient)
        bias2 = (0.99 * bias2) + (.01 * bias2Gradient)
        bias1 = (0.99 * bias1) + (.01 * bias1Gradient)
        
        #print("errorGradient = ", errorGradient)
        xs.append(inputs[0][i])
        ys.append(outputs[0][i])
    
    
    if _ == 49:
        print(xs)
        print(ys)
        plt.plot(xs,ys)
        plt.pause(0.1)
    

plt.show()
