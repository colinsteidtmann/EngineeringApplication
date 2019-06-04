import numpy as np
import matplotlib.pyplot as plt
import network

class NeuralNetwork:
    def __init__(self):
        self.num_neurons = 4
        self.weightLayer2 = np.random.normal(-1,1,(self.num_neurons, 1))
        self.biasLayer2 = np.random.normal(0,1,(self.num_neurons,1))
        self.weightLayer3 = np.random.normal(-1,1,(1, self.num_neurons))
        self.biasLayer3 = np.random.normal(0,1,(1, 1))
    
    def runBatch(self, Xs, Ys):
        # self.weightLayer3UpdateArray = []
        # self.weightLayer2UpdateArray = []
        # self.biasLayer3UpdateArray = []
        # self.biasLayer2UpdateArray = []
        
        predictionYs = []
        for x, y in zip(Xs, Ys):
            prediction = self.forwards(x)
            self.backwards(y)
            predictionYs.append(prediction[0])
        
        # avgWeightLayer3Update = sum(self.weightLayer3UpdateArray) #/ len(self.weightLayer3UpdateArray)
        # avgWeightLayer2Update = sum(self.weightLayer2UpdateArray) #/ len(self.weightLayer2UpdateArray)
        # avgBiasLayer3Update = sum(self.biasLayer3UpdateArray) #/ len(self.biasLayer3UpdateArray)
        # avgBiasLayer2Update = sum(self.biasLayer2UpdateArray)  #/ len(self.biasLayer2UpdateArray)
        # batch_size = len(self.weightLayer3UpdateArray)
        #print(avgWeightLayer3Update)
        # self.weightLayer3 -= ((.0001) * avgWeightLayer3Update)
        # self.weightLayer2 -= ((.0001) * avgWeightLayer2Update)
        # self.biasLayer3 -= ((.0001) * avgBiasLayer3Update)
        # self.biasLayer2 -= ((.0001) * avgBiasLayer2Update)

        return predictionYs

    def deriv_of_relu(self, layer):
        layer[layer <= 0] = 0.01
        layer[layer > 0] = 1
        return layer

    def forwards(self, number):
        self.aLayer1 = np.array([number]).reshape((1, 1))
        self.zLayer2 = np.matmul(self.weightLayer2, self.aLayer1) + self.biasLayer2
        self.aLayer2 = np.maximum(0.01*self.zLayer2, self.zLayer2)
        self.zLayer3 = np.matmul(self.weightLayer3, self.aLayer2) + self.biasLayer3
        self.aLayer3 = self.zLayer3[0]

        return self.aLayer3
    
    def backwards(self, label):
        delta_layer3 = np.array([(self.aLayer3 - label) * 1]).reshape((1, 1))
        delta_layer2 = np.matmul(self.weightLayer3.reshape((self.num_neurons, 1)), delta_layer3) * self.deriv_of_relu(self.zLayer2)
        weightLayer3Update = np.matmul(delta_layer3, self.aLayer2.reshape((1, self.num_neurons)))
        weightLayer2Update = np.matmul(delta_layer2, self.aLayer1.reshape((1, 1)))
        biasLayer3Update = delta_layer3
        biasLayer2Update = delta_layer2

        #print("weightLayer3Update = ", weightLayer3Update)
        # print("weightLayer2Update = ", weightLayer2Update)
        # print("biasLayer3Update = ", biasLayer3Update)
        # print("biasLayer2Update = ", biasLayer2Update)

        # self.weightLayer3UpdateArray.append(weightLayer3Update)
        # self.weightLayer2UpdateArray.append(weightLayer2Update)
        # self.biasLayer3UpdateArray.append(biasLayer3Update)
        # self.biasLayer2UpdateArray.append(biasLayer2Update)
        
        self.weightLayer3 -= (.01 * weightLayer3Update)
        self.weightLayer2 -= (.01 * weightLayer2Update)
        self.biasLayer3 -= (.01 * biasLayer3Update)
        self.biasLayer2 -= (.01 * biasLayer2Update)
        





nn = network.Network(sizes=[1,64,64,1],cost="quadratic",activation="leaky_relu", optimizer="adam", dropout=False) 
plt.ion()
fig, ax = plt.subplots()

x = np.arange(0, np.pi * 2 , 0.1)
y = np.sin(x)
ax.plot(x, y)
ax.scatter(x, y)

predictions = np.zeros(len(x))
sc = ax.scatter(x, predictions)
pc, = ax.plot(x,predictions)
plt.xlim(0,20)
plt.ylim(-2,4)

plt.draw()

for training in range(50):
    predictions = np.squeeze(nn.feedforward(x))
    nn.SGD(training_data=zip(x, y), lr=.001, mini_batch_size=128, epochs=45, updateLoss=True)
    print("episode = ", training, "episodeLoss = ", nn.episodeLoss)
    sc.set_offsets(np.c_[x, predictions])
    pc.set_ydata(predictions)
    fig.canvas.draw_idle()
    plt.pause(0.1)

print(nn.weights)
print(nn.biases)
predictions = np.squeeze(nn.feedforward(x))
plt.plot(x, predictions)
plt.show()

