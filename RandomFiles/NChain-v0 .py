import gym
import numpy as np
import random
class NeuralNetwork:
    def __init__(self,env):
        self.weights = np.random.rand(5, 16)
        self.weights2 = np.random.rand(16,2)
    
    def one_hot_state(self, state):
        one_hot_array = np.zeros((1,5))
        one_hot_array[0][state] = 1
        return one_hot_array

    def relu(self, weightMatrix):
        for weight in weightMatrix[0]:
            weight = max(0,weight)
        return weightMatrix
    
    def forward(self, state):
        actionsAndValues = []
        hiddenLayer1 = 0
        for i in range(6):
            hiddenLayer1 = np.dot(state, self.weights)
            dropoutLayer = np.random.binomial([np.ones((1, 16))], 1 - 0.2)[0] * (1.0 / (1 - 0.2))
            hiddenLayer1 *= dropoutLayer
            hiddenLayer1 = self.relu(hiddenLayer1)
            outputLayer = np.dot(hiddenLayer1, self.weights2)
            action = np.argmax(outputLayer)
            value = outputLayer[0][action]
            
            actionsAndValues.append([action,value])
        
        variance = np.var([x[1] for x in actionsAndValues])
        print("variance = ", variance)
        sample = random.sample(actionsAndValues, 1)[0]
        action = sample[0]
        value = sample[1]
        
        return variance, action, value, hiddenLayer1

    def backprop(self, state, action, variance, value, reward, new_state, done, hiddenLayer1):
        new_state = self.one_hot_state(new_state)
        variance, action, next_value, _ = self.forward(new_state)

        if not done:
            q_value = reward + .95 * next_value
        else:
            q_value = reward
        error = (value - q_value) * -np.log(variance)

        print("loss = ", .1 * error, "xxx = ", -np.log(variance))
        weight2Gradient = error * hiddenLayer[]
        weight1Gradient = error * value * hiddenLayer1[]
        #print("weight2Gradient, weight1Gradient = ", weight2Gradient, weight1Gradient)
        self.weights2 -= (.1 * error)
        self.weights -= (.1 * error)
        

            

env = gym.make("NChain-v0")   
nn = NeuralNetwork(env)
reward_total = 0
for episode in range(5):
    state = env.reset()
    while True:
        #env.render()
        state_input = nn.one_hot_state(state)
        variance, action, value, hiddenLayer1 = nn.forward(state_input)
        new_state, reward, done, _ = env.step(action)    
        nn.backprop(state, action, variance, value, reward, new_state, done, hiddenLayer1)
        reward_total += reward 
        state = new_state
        if done:
            break
    
    if episode % 500 == 0:
        print("episode avg = ", reward_total/500, "episode = ", episode)
        reward_total = 0