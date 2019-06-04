import gym
import numpy as np
import random
class NeuralNetwork:
    def __init__(self, env):
        self.number = 0.
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.weights1 = np.random.rand(self.num_states, self.num_states)
        self.weights2 = np.random.rand(self.num_states, self.num_actions)
        self.hiddenLayer = 0

    def one_hot_state(self, state):
        one_hot_array = np.zeros((1,self.num_states))
        one_hot_array[0][state] = 1
        return one_hot_array

    def relu(self, weights):
        for weight in weights[0]:
            weight = max(0,weight)
        return weights

    def forward(self,state):
        self.hiddenLayer = np.dot(state, self.weights1)
        self.hiddenLayer = self.relu(self.hiddenLayer)
        output_num = np.dot(self.hiddenLayer, self.weights2)[0]        
        return output_num

    def backprop(self, state, reward):
        error = (1-reward)
        weights2Gradient = error * self.hiddenLayer
        weights1Gradient = error * self.weights2 * state
        self.weights2 -= (.01*weights2Gradient).reshape(self.num_states, self.num_actions)
        self.weights1 -= (.01*weights1Gradient).reshape(self.num_states, self.num_states)
        

env = gym.make("HotterColder-v0")          
nn = NeuralNetwork(env)
for episode in range(1):
    state = env.reset()
    reward = 0
    while reward < .999:
        print("state = ", state)
        state = nn.one_hot_state(state)
        prediction = nn.forward(state)
        new_state, reward, done, info = env.step(prediction)
        nn.backprop(state, reward)
        state = new_state
        print("reward = ", reward)
