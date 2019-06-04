import gym
import numpy as np
import random
import warnings
class NeuralNetwork:
    def __init__(self,env):
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.neuronSize = 64
        self.weights1 = np.random.rand(self.num_states, self.neuronSize)
        self.weights2 = np.random.rand(self.neuronSize, self.num_actions)
        self.epsilon = .99
        self.errors = []
        self.memory = []

        self.weightsMemory = [[],[]]

    def relu(self, weights):
            x = np.maximum(weights,.05*weights)
            return x

    def approx(self, state, action):
        hiddenLayer = np.dot(state, self.weights1)
        hiddenLayer = self.relu(hiddenLayer)
        outputValues = np.dot(hiddenLayer, self.weights2)
        value = outputValues[0][action]
        self.weightsMemory[action] = (np.copy(self.weights1),hiddenLayer,np.copy(self.weights2),value)
        return value
    
    def forward(self, state):
        value_array = [self.approx(state,0),self.approx(state,1)]
        action = np.argmax(value_array)
        self.epsilon *= .99999
        if np.random.rand() < self.epsilon:
            action = env.action_space.sample()
        if self.epsilon < .1:
            self.epsilon = .1
        return action

    def backprop(self, state, action, weightsMemory, reward, new_state, done):
        new_state = np.array(new_state).reshape((1, 4))
        if not done:
            q_value = reward + .95 * self.approx(new_state, self.forward(new_state))
        else:
            q_value = reward
        #print("q_value = ", q_value)
        # print("value = ", weightsMemory[action][3], "q_value = ", q_value)
        error = (weightsMemory[action][3] - q_value)
        self.errors.append(error)
        error_matrix = np.array([0., 0.]).reshape((1,2))
        error_matrix[0][action] = error
        weights2Gradient=weightsMemory[action][1].reshape((64, 1)).dot(error_matrix)
        weights1Gradient=(np.maximum(state, .05 * state).reshape((4, 1)).dot(error_matrix)).dot(weightsMemory[action][2].reshape((2, 64)))
        if (weights2Gradient[0][0] > 10 or weights1Gradient[0][0] > 10 or  weights2Gradient[0][0] < -10 or weights1Gradient[0][0] < -10):
            print("error = ", error, "weights2Gradient[0][0] = ", weights2Gradient[0][0], "weights1Gradient[0][0] = ", weights1Gradient[0][0])
        self.weights2 -= (.001 *weights2Gradient)
        self.weights1 -= (.001 * weights1Gradient)

        
        

            

env = gym.make("CartPole-v1")          
nn = NeuralNetwork(env)
rewards = []
for episode in range(1000000):
    state = env.reset()
    reward_total = 0
    while True:
        #env.render()
        state = np.array(state).reshape((1,4))
        action = nn.forward(state)
        new_state, reward, done, _ = env.step(action)
        stateCopy = np.copy(nn.weightsMemory)
        nn.backprop(state, action,stateCopy, reward, new_state, done)
        reward_total += reward
        state = new_state
        if done:
            rewards.append(reward_total)
            break
        
    
    if episode % 500 == 0:
        print("episode avg = ", sum(rewards)/len(rewards),"avg_error = ",sum(nn.errors)/len(nn.errors), "episode = ", episode)
        rewards = []
        nn.errors = []