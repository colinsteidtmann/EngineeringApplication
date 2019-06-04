import gym
import numpy as np
import random

class NeuralNetwork:
    def __init__(self,env):
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.value_table = np.zeros((self.num_states, self.num_actions))
        self.lr = 0.1
        self.gamma = .95

    
    def predict_value(self, state):
        return np.max(self.value_table[state])

    def choose_action(self, state):
        return np.argmax(self.value_table[state])

    def update_value_table(self, state, action, reward,done, new_state):
        if done:
            value_update = reward
        else:
            value_update = reward + self.gamma * self.predict_value(new_state)
        
        self.value_table[state][action] = (1-self.lr)*self.value_table[state][action] + (self.lr)*value_update

            

env = gym.make("FrozenLake-v0")          
nn = NeuralNetwork(env)
reward_total = 0
for episode in range(50000):
    state = env.reset()
    choices = []
    while True:
        #env.render()
        action = nn.choose_action(state)
        new_state, reward, done, _ = env.step(action)
        if done and reward != 1:
            reward -= .5
        nn.update_value_table(state, action, reward,done, new_state)
        reward_total += reward 
        state = new_state
        if done:
            break
    
    if episode % 500 == 0:
        print("episode avg = ", reward_total/500, "episode = ", episode)
        reward_total = 0

print("value_table = ", nn.value_table)