import gym
import numpy as np
import random
class NeuralNetwork:
    def __init__(self):
        self.upper = 1000.
        self.lower = -1000.
        self.guess = 0
    
    def forward(self):
        self.guess = (self.upper+self.lower)/2
        return self.guess

    def backprop(self, new_state):
        if new_state == 1:
            self.lower = self.guess
        else:
            self.upper = self.guess
            
nn = NeuralNetwork()
env = gym.make("GuessingGame-v0")
upper = 1000
lower = -1000
for episode in range(1):
    state = env.reset()
    reward = 0
    i = 0
    while reward < 1:
        guess = nn.forward()
        guess = np.array(guess).reshape(1)
        new_state, reward, done, info = env.step(guess)
        nn.backprop(new_state)
        state = new_state
        i+=1
        print("reward = ", reward, "i = ", i)
