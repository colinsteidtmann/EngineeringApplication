import numpy as np
import SharedFunctions as sf
import ConvNet as cnn
import FullyConnected as fcn
import NeuralNetwork as nen
import origNN as ofcn
import datetime
import copy

orig_fully_connected = ofcn.NeuralNetwork(sizes=[3136, 1024, 1],activations=["linear", "relu", "linear"], scale_method="normalize", optimizer="nadam", lr=.01, lr_decay=0.0)
fully_connected = fcn.FullyConnected(sizes=[32, 4, 10],activations=["linear", "relu", "linear"], scale_method="normalize", optimizer="nadam", lr=.01, lr_decay=0.0)
convnet = cnn.ConvNet(
            conv_method="convolution",
            layer_names=["conv", "pool", "conv", "pool"],
            num_filters=[3, None, 2, None],
            kernel_sizes=[[2,2], None, [2,2], None],
            stride_sizes=[[2,2], [1,1], [2,2], [1,1]],
            pool_sizes=[None, [2,2], None, [2,2]],
            pool_fns=[None, "max", None, "max"],
            pad_fns=["same", "valid", "same", "valid"],
            activations=["relu", None, "relu", None],
            input_channels=2,
            scale_method="normalize",
            optimizer="nadam",
            lr=0.01,
            lr_decay=0
        )

nn = nen.NeuralNetwork([convnet, fully_connected], "cross_entropy")

# inputs = np.random.rand(2, 6, 6, 2)
# #print(nn.feedforward(inputs, scale=True))

# epsilon=1e-5
# inputs = np.random.rand(2, 6, 6, 2)
# outputs = nn.feedforward(inputs)
# print(outputs.shape)
# epsilon = 1e-5
# a = np.array([9.9, 8.2]).reshape((1,2))
# y = np.array([8,5]).reshape((1,2))
# loss = nn.loss(a, y)
# plus_a = copy.deepcopy(a)
# plus_a[0,1] += epsilon
# plus_out = nn.loss(plus_a, y)
# minus_a = copy.deepcopy(a)
# minus_a[0,1] -= epsilon
# minus_out = nn.loss(minus_a, y)
# approx_grad = (plus_out - minus_out) / (2 * epsilon)
# calc_grad = nn.loss_prime(a, y)
# print("calc_grad = ", calc_grad)
# print("approx_grad = ", approx_grad)
# kernels = np.array(convnet.get_weights())
# plus_kernels = copy.deepcopy(kernels)
# minus_kernels = copy.deepcopy(kernels)
# plus_kernels[2][0, 0, 0, 0]  += epsilon
# minus_kernels[2][0, 0, 0, 0] -= epsilon

# convnet.set_weights(plus_kernels)
# plus_out = nn.feedforward(inputs)
# convnet.set_weights(minus_kernels)
# minus_out = nn.feedforward(inputs)
# convnet.set_weights(kernels)
# approx_grad = ((plus_out - minus_out) / (2 * epsilon))
# print(approx_grad)

# calc_grad = nn.gradients(inputs, [0], "weights")
# #print(plus_out.shape)
# print(calc_grad[0][0][1][:,0, 0, 0, 0])
# print(calc_grad[0][0][0][:,2, 1, 1, 1])
# print(approx_grad*np.array([2,3]).reshape(approx_grad.shape))
"""it all works, add more padding things now, clean up check gradients """
#grads = nn.gradients(inputs, [0,1], ["zLayer", "weights", "biases"], None)
#nn.check_gradients(inputs)
# labels = [4, 2, 3, 4]
# cost = nn.get_cost(inputs, labels)
# nn.gradients(inputs, 1, "weights")
# nn.check_gradients(inputs)
# start = datetime.datetime.now()
# for _ in range(10):
#     conv_inputs = np.random.rand(1, 28, 28, 3)
#     conv_outputs = convnet.feedforward(conv_inputs, scale=True)
#     conv_outputs2 = conv_outputs.reshape((1, -1))
#     dense_outputs = fully_connected.feedforward(conv_outputs2)
#     dense_gradients = fully_connected.gradients(conv_outputs2, None, 1, 3, "zLayer", "output", None)
#     dense_gradients = dense_gradients.reshape(conv_outputs.shape)
#     conv_gradients = convnet.gradients(conv_inputs, [2, 4], ["weights", "biases"], dense_gradients)
#     print(dense_gradients.shape)




#     # derivs = [np.random.rand(1, 7, 7, 64) for _ in range(10)]
#     # derivs = np.array(derivs, ndmin=4)
#     # gradients = convnet.gradients(inputs, [2, 4], ["weights", "biases"], derivs)
#     # gradients2 = convnet.gradients(inputs, [1], "zLayer", derivs)
#     # print(gradients2.shape)
#     # nablaWs, nablaBs = [gradients[0]], [gradients[1]]
#     # convnet.apply_gradients(zip(nablaWs, nablaBs))
# finish = datetime.datetime.now()
# print("time: ", finish-start)



# # start = datetime.datetime.now()
# # for _ in range(100):
# #     inputs = np.random.rand(1, 3) 
# #     outputs1 = orig_fully_connected.feedforward(inputs, scale=True)
# #     outputs = fully_connected.feedforward(inputs, scale=True)
# #     gradients1 = fully_connected.gradients(inputs, None, np.arange(2, 5), 4, "weights", "output", None)
# #     gradients2 = fully_connected.gradients(inputs, None, np.arange(2, 5),4, "biases", "output", None)
# #     nablaWs, nablaBs = gradients1, gradients2
# #     fully_connected.apply_gradients(zip(nablaWs, nablaBs))
# #     nablaWs = [orig_fully_connected.gradients(input1, None, np.arange(2, 5),4, "weights", "output", None) for input1 in inputs]
# #     nablaBs = [orig_fully_connected.gradients(input1, None, np.arange(2, 5),4, "biases", "output", None) for input1 in inputs]
# #     orig_fully_connected.apply_gradients(zip(nablaWs, nablaBs))
# # finish = datetime.datetime.now()
# # print("time: ", finish - start)

