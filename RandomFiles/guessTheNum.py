import numpy as np

class book:
    def __init__(self):
        self.weights1 = np.random.rand(1,64)
        self.weights2 = np.random.rand(64,1)
        self.hiddenlayer = 0
        self.label = 8

    def relu(self, weights):
        for weight in weights[0]:
            weight = max(0,weight)
        return weights

    def forward(self, input):
        self.hiddenlayer = np.dot(input,self.weights1)
        self.hiddenlayer = self.relu(self.hiddenlayer)
        print("hiddenlayer = ", self.hiddenlayer)
        output = np.dot(self.hiddenlayer,self.weights2)[0][0]
        return output

    def backprop(self,output):
        error = (output - self.label)
        weights1Gradient = error*
        self.weights1 -= (.001 * input_neuron_grad).reshape((1,2))
        self.weights2 -= (.001*output_neuron_grad).reshape((2,1))

input = np.array([2]).reshape((1,1))
book = book()
for i in range(10000):
    out = book.forward(input)
    book.backprop(out)
    print("out = ", out)
