from mlxtend.data import loadlocal_mnist
import sys
sys.path.append("..")
import random
import gym
import math
import numpy as np
from collections import deque
from NeuralNets import ConvNet as cnn
from NeuralNets import FullyConnected as fcn
from NeuralNets import NeuralNetwork as neuralnetwork
fully_connected = fcn.FullyConnected(sizes=[800, 120, 10], activations=["relu", "relu", "softmax"], scale_method=None, optimizer="nadam", lr=.001, lr_decay=(0.0))
fully_connected.add_dropout([1], 0.4)
convnet = cnn.ConvNet(
            conv_method="convolution",
            layer_names=["conv", "pool", "conv", "pool"],
            num_filters=[16,None,32,None],
            kernel_sizes=[[5,5],None,[5,5],None],
            stride_sizes=[[1,1],[2,2],[1,1],[2,2]],
            pool_sizes=[None,[2,2],None,[2,2]],
            pool_fns=[None,"max",None,"max"],
            pad_fns=["same","valid","valid","valid"],
            activations=["relu",None,"relu",None],
            input_channels=1,
            scale_method=None,
            optimizer="nadam",
            lr=0.001,
            lr_decay=(0.0)
        )

nn = neuralnetwork.NeuralNetwork([convnet, fully_connected], loss_fn="cross_entropy")

images, labels = loadlocal_mnist(
        images_path='/Users/colinsteidtmann/openai/Mnist/train-images-idx3-ubyte', 
        labels_path='/Users/colinsteidtmann/openai/Mnist/train-labels-idx1-ubyte')
inputs = images.reshape((-1, 28, 28, 1))
labels_one_hot = np.zeros((len(labels), 10))
for idx, hot_idx in enumerate(labels):
    labels_one_hot[idx,hot_idx] = 1

nn.sgd_fit(inputs, labels_one_hot, batch_size=125, epochs=1, train_pct=0.8, shuffle_inputs=False)