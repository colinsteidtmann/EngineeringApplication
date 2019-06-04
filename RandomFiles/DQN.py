import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import sys

class NeuralNetwork:
    def __init__(self, env):
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        self.onlineInputWeights = np.random.rand(self.num_states, 16)
        self.onlineInputBias = np.random.rand(1,16)
        self.onlineOutputWeights = np.random.rand(16, self.num_actions)
        self.onlineOutputBias = np.random.rand(1,2)
        self.targetInputWeights = self.onlineInputWeights[:]
        self.targetInputBias =  self.onlineInputBias[:]
        self.targetOutputWeights = self.onlineOutputWeights[:]
        self.targetOutputBias = self.onlineOutputBias[:]

        self.target_update_time = 32
        self.epsilon = 1
        self.epsilon_decay_rate = 0.99999375
        self.gamma = .95
        self.lr = .1
        self.replay_buffer = []
        

    def choose_action(self, state):
        _, _, _, _,_,predicted_values = self.live_predict(state)
        if (np.random.rand() < self.epsilon):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(predicted_values)
        self.epsilon *= self.epsilon_decay_rate
        if self.epsilon < .1:
            self.epsilon = .1
        return action

    def update_replay(self, state, action, reward, new_state, done):
        if len(self.replay_buffer) > 1000:
            del self.replay_buffer[0]
        self.replay_buffer.append([state, action, reward, new_state, done])

        if len(self.replay_buffer) > 32:
            batch = random.sample(self.replay_buffer, 32)
            self.train(batch)

    def relu(self,weights):
        return np.maximum(weights, 0)

    def live_predict(self, state):
        hidden_layer = np.dot(state, self.onlineInputWeights) + self.onlineInputBias
        relu_hidden_layer = self.relu(hidden_layer)
        actionValues = np.dot(relu_hidden_layer, self.onlineOutputWeights) + self.onlineOutputBias
        return self.onlineInputWeights[:], self.onlineInputBias[:], relu_hidden_layer[:], self.onlineOutputWeights[:], self.onlineOutputBias[:], actionValues[0]

    def target_predict(self, state):
        hidden_layer = np.dot(state, self.targetInputWeights) + self.targetInputBias
        relu_hidden_layer = self.relu(hidden_layer)
        actionValues = np.dot(relu_hidden_layer, self.targetOutputWeights) + self.targetOutputBias
        return self.targetInputWeights[:], self.targetInputBias[:], relu_hidden_layer[:], self.targetOutputWeights[:], self.targetOutputBias[:], actionValues[0]
    
    
    def get_derivtive_of_error(self, target_value, online_value):
        if np.abs(target_value-online_value) <= 1:
            deriv = (target_value - online_value)
        else:
            deriv = (target_value-online_value)/np.abs(target_value-online_value)
        return deriv

    def get_derivtive_of_hidden_layer(self, hl):
        hl[hl <= 0] = 0
        hl[hl > 0] = 1
        return hl

    def update_target_network(self):
        self.targetInputWeights = self.onlineInputWeights[:]
        self.targetInputBias =  self.onlineInputBias[:]
        self.targetOutputWeights = self.onlineOutputWeights[:]
        self.targetOutputBias = self.onlineOutputBias[:]


    def train(self, batch):
        for state, action, reward, new_state, done in batch:
            if done:
                target_value = reward
            else:
                _,_,_,_,_,value = self.target_predict(new_state)
                target_value = reward + self.gamma*np.max(value)
            
            
            weights1, bias1, relu_hiddenLayer, weights2, bias2, online_values = self.live_predict(state)
            deriv_of_error = np.array([self.get_derivtive_of_error(target_value, online_values[action])]).reshape((1,1))
            deriv_of_hiddenlayer = self.get_derivtive_of_hidden_layer(np.dot(state,weights1))

            # print("weights1Gradient.shape = ", online_values[action])
            # sys.exit()
            

            weights1Gradient = np.dot(deriv_of_error,weights2[:,action].reshape((1,16)))
            weights2Gradient = np.dot(relu_hiddenLayer.reshape((16, 1)), deriv_of_error)
            bias1Gradient = np.dot(deriv_of_error, bias1)
            bias2Gradient = np.dot(deriv_of_error,bias2)

            self.onlineOutputWeights[:, action] = ((1 - self.lr) * self.onlineOutputWeights[:, action]) + (self.lr * weights2Gradient).reshape(16)
            self.onlineInputWeights = (1 - self.lr) * self.onlineInputWeights + self.lr * weights1Gradient
            self.onlineInputBias = (1 - self.lr) * self.onlineInputBias + self.lr * bias1Gradient
            self.onlineOutputBias = (1-self.lr) * self.onlineOutputBias + self.lr*bias2Gradient
            


env = gym.make("CartPole-v1")          
nn = NeuralNetwork(env)
reward_total = 0
for episode in range(50000):
    state = env.reset()
    while True:
        #env.render()
        state = np.array(state).reshape((1,nn.num_states))
        action = nn.choose_action(state)
        new_state, reward, done, _ = env.step(action)
        new_state = np.array(new_state).reshape((1,nn.num_states))
        nn.update_replay(state, action, reward, new_state, done)
        reward_total += reward 
        state = new_state
        if done:
            break
    
    if episode % 500 == 0:
        print("episode avg = ", reward_total / 500, "episode = ", episode)
        print("old weights1 = ", nn.targetInputWeights)
        print("new weights1 = ", nn.onlineInputWeights)
        reward_total = 0
    
    if episode % nn.target_update_time == 0:
        
        nn.update_target_network()
