import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        self.weightLayer2 = np.random.rand(10, 4)
        self.biasLayer2 = np.random.rand(10,1)
        self.weightLayer3 = np.random.rand(1, 10)
        self.biasLayer3 = np.random.rand(1, 1)
    
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
        
        # avgWeightLayer3Update = sum(self.weightLayer3UpdateArray) / len(self.weightLayer3UpdateArray)
        # avgWeightLayer2Update = sum(self.weightLayer2UpdateArray) / len(self.weightLayer2UpdateArray)
        # avgBiasLayer3Update = sum(self.biasLayer3UpdateArray) / len(self.biasLayer3UpdateArray)
        # avgBiasLayer2Update = sum(self.biasLayer2UpdateArray) / len(self.biasLayer2UpdateArray)

        # self.weightLayer3 -= (.001 * avgWeightLayer3Update)
        # self.weightLayer2 -= (.001 * avgWeightLayer2Update)
        # self.biasLayer3 -= (.001 * avgBiasLayer3Update)
        # self.biasLayer2 -= (.001 * avgBiasLayer2Update)

        return predictionYs

    def deriv_of_relu(self, layer):
        layer[layer <= 0] = 0.1
        layer[layer > 0] = 1
        return layer

    def forwards(self, number):
        self.aLayer1 = np.array([number]).reshape((4, 1))
        self.zLayer2 = np.matmul(self.weightLayer2, self.aLayer1) + self.biasLayer2
        self.aLayer2 = np.maximum(0.1*self.zLayer2, self.zLayer2)
        self.zLayer3 = np.matmul(self.weightLayer3, self.aLayer2) + self.biasLayer3
        self.aLayer3 = self.zLayer3[0]

        return self.aLayer3
    
    def backwards(self, label):
        delta_layer3 = np.array([(self.aLayer3 - label) * 1]).reshape((4, 1))
        delta_layer2 = np.matmul(self.weightLayer3.reshape((10, 1)), delta_layer3) * self.deriv_of_relu(self.zLayer2)
        weightLayer3Update = np.matmul(delta_layer3, self.aLayer2.reshape((1, 10)))
        weightLayer2Update = np.matmul(delta_layer2, self.aLayer1.reshape((1, 1)))
        biasLayer3Update = delta_layer3
        biasLayer2Update = delta_layer2

        # print("weightLayer3Update = ", weightLayer3Update)
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
        








x = np.random.rand(50,4)
y = np.ones((50,1))
nn = NeuralNetwork()
plt.plot(x, y)
plt.scatter(x, y)

for guess in range(100):
    predictionYs = nn.runBatch(x, y)
    plt.plot(x, predictionYs)
    plt.pause(0.1)

plt.show()