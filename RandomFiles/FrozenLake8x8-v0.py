import gym
import numpy as np
import random
class NeuralNetwork:
    def __init__(self,env):
        self.weights = np.zeros((64,4))
        self.epsilon = .99
    
    def one_hot_state(self, state):
        one_hot_array = np.array(np.zeros((1,64)))
        one_hot_array[0][state] = 1
        return one_hot_array
    
    def forward(self, state):
        step_forwards = np.matmul(state, self.weights)
        action = np.argmax(step_forwards)
        
        self.epsilon *= .999999
        if np.random.rand() < self.epsilon:
            action = env.action_space.sample()
        
        return action

    def backprop(self, state, action, reward, new_state, done):
        if not done:
            q_value = reward + .95 * np.max(self.weights[new_state])
        else:
            q_value = reward
        loss = (self.weights[state][action] - q_value)
        self.weights[state][action] -= .1 * loss
        

            

env = gym.make("FrozenLake8x8-v0")          
nn = NeuralNetwork(env)
reward_total = 0
for episode in range(500000):
    state = env.reset()
    while True:
        #env.render()
        state_input = nn.one_hot_state(state)
        action = nn.forward(state_input)
        new_state, reward, done, _ = env.step(action)    
        nn.backprop(state, action, reward, new_state, done)
        reward_total += reward 
        state = new_state
        if done:
            break
    
    if episode % 500 == 0:
        print("episode avg = ", reward_total/500, "episode = ", episode)
        reward_total = 0