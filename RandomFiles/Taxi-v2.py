#implemented with SARSA
import gym
import numpy as np
import random
class NeuralNetwork:
    def __init__(self,env,num_actions,num_states):
        self.weights = np.random.rand(500,6)
        self.epsilon = .99
    
    def one_hot_state(self, state):
        one_hot_array = np.array(np.zeros((1,500)))
        one_hot_array[0][state] = 1
        return one_hot_array
    
    def forward(self, state):
        step_forwards = np.matmul(state, self.weights)
        action = np.argmax(step_forwards)
        
        self.epsilon *= .999999
        if np.random.rand() < self.epsilon:
            action = env.action_space.sample()
        
        return action

    def backprop(self, state, action, reward, new_state,action2, done):
        if not done:
            q_value = reward + .95 * self.weights[new_state][action2]
        else:
            q_value = reward
        loss = (self.weights[state][action] - q_value)
        self.weights[state][action] -= .1 * loss
        

            

env = gym.make("Taxi-v2")
num_actions = env.action_space.shape[0]
num_states = env.observation_space.shape[0]
#print("num_actions, num_states = ", num_actions, num_states)             
nn = NeuralNetwork(env, num_actions, num_states)

reward_total = 0
for episode in range(500000):
    state = env.reset()
    action = nn.forward(nn.one_hot_state(state))
    while True:
        #env.render()
        
        new_state, reward, done, _ = env.step(action)
        action2 = nn.forward(nn.one_hot_state(new_state))  
        nn.backprop(state, action, reward, new_state,action2, done)
        reward_total += reward 
        state = new_state
        action = action2
        if done:
            break
    
    if episode % 500 == 0:
        print("episode avg = ", reward_total/500, "episode = ", episode)
        reward_total = 0