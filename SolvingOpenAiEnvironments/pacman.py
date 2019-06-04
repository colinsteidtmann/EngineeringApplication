# Inspired by https://keon.io/deep-q-learning/
import sys
sys.path.append("..")
import random
import gym
import math
import numpy as np
import cv2
import sys
from NeuralNets import ConvNet as cnn
from NeuralNets import FullyConnected as fcn
from NeuralNets import NeuralNetwork as neuralnetwork
from collections import deque


""" Examples of outputs (self-reference guide):
    model.feedforward([input]) --> [output]
    model.gradients(x,y,1,4,"input","loss") --> [array([[-0.46799199],
                                                [ 0.11585874],
                                                [ 0.09831851],
                                                [ 0.44538445]])]
    np.concatenate(([1,2,3],[4]), axis=0) --> array([1, 2, 3, 4])
    np.concatenate(([[1,2,3]],[[4]]), axis=1) --> array([[1, 2, 3, 4]])
"""


class DQNCartPoleSolver():
    def __init__(self, n_episodes=2000, min_reward=3000, max_env_steps=10000, gamma=0.99, exploration_steps=1000, batch_size=100, quiet=False):
        self.memory = deque(maxlen=1000000)
        self.env = gym.make('MsPacman-v0')
        self.gamma = gamma
        self.critic_alpha = 0.01
        self.actor_alpha = 0.001
        self.critic_alpha_decay = 0.001
        self.actor_alpha_decay = 0.0
        self.n_episodes = n_episodes
        self.min_reward = min_reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.max_episode_steps = max_env_steps
        self.exploration_steps = exploration_steps
        self.batch_size = batch_size
        self.tau = 0.1
        self.quiet = quiet
        self.states = []

        # Init models
        self.init_models()

    
    def output_scale(self, x):
        # return np.squeeze(np.clip(np.array(x),-2,2))
        return np.squeeze(np.array(x) * 2)
    
    def output_scale_deriv(self, z):
        # z[z <= -2 or z >= 2] = 0
        # z[z > -2 and z < 2] = 1
        # return z
        return np.array(1)
        
    def init_models(self):
        self.fully_connected = fcn.FullyConnected(sizes=[441, 120, 9], activations=["relu", "relu", "linear"], scale_method=None, optimizer="nadam", lr=.001, lr_decay=(0.0))
        self.fully_connected.add_dropout([1], 0.4)
        self.convnet = cnn.ConvNet(
                    conv_method="convolution",
                    layer_names=["conv", "pool", "conv", "pool", "conv", "pool"],
                    num_filters=[3,None,6,None,9,None],
                    kernel_sizes=[[5,5],None,[5,5],None,[5,5],None],
                    stride_sizes=[[1,1],[2,2],[1,1],[2,2],[1,1],[2,2]],
                    pool_sizes=[None,[2,2],None,[2,2],None,[2,2]],
                    pool_fns=[None,"max",None,"max",None,"max"],
                    pad_fns=["same","valid","valid","valid","valid","valid"],
                    activations=["relu",None,"relu",None,"relu",None],
                    input_channels=4,
                    scale_method=None,
                    optimizer="nadam",
                    lr=0.001,
                    lr_decay=(0.0)
                )
        self.model = neuralnetwork.NeuralNetwork([self.convnet, self.fully_connected], loss_fn="mean_squared_error")
        
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, step, test=False):
        if (np.random.random() <= self.epsilon):
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.model.feedforward(state, test=test))
        return action
    
    def preprocess_state(self, state):
        state = cv2.cvtColor(state.reshape((210, 160, 3))[0:160], cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (84,84))
        while len(self.states) < 4:
            self.states.append(state)
        self.states.pop(0)
        self.states.append(state)
        return np.reshape(self.states, (1,84,84,4))
        

        #gray = cv2.cvtColor(state.reshape((210, 160, 3))[0:160], cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        
        batch_states = np.array([i[0] for i in minibatch]).reshape((len(minibatch), 84, 84, 4))
        batch_actions = np.array([i[1] for i in minibatch]).reshape((len(minibatch),1,1))
        batch_rewards = np.array([i[2] for i in minibatch]).reshape((len(minibatch), 1, 1))
        batch_next_states = np.array([i[3] for i in minibatch]).reshape((len(minibatch), 84, 84, 4))
        batch_dones = np.array([i[4] for i in minibatch]).reshape((len(minibatch), 1, 1))
        random_states = self.model.get_nprandomstates()
        y_target = self.model.feedforward(batch_states)
        target = (batch_rewards + (np.invert(batch_dones) * (self.gamma * np.amax(self.model.feedforward(batch_next_states), 1, keepdims=True)))).reshape((len(minibatch), 1, 1))
        np.put_along_axis(y_target, batch_actions, target, 1)

    
        self.model.sgd_fit(batch_states, y_target, batch_size=batch_size, epochs=1, train_pct=1.0, shuffle_inputs=False, random_states=random_states)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run_one_episode(self, total_steps):
        state = self.preprocess_state(self.env.reset())
        done = False
        rewards = 0
        while not done:
            #self.env.render()
            action = self.choose_action(state, total_steps, test=True)
            next_state, reward, done, _ = self.env.step(action)
            next_state = self.preprocess_state(next_state)
            state = next_state
            rewards += reward
        return rewards

    def run(self):
        scores = deque(maxlen=100)
        total_steps = 1
        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            steps = 0
            while not done:
                action = self.choose_action(state, total_steps)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += reward
                total_steps += 1
                steps += 1
                if done:
                    break

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.min_reward and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials'.format(e, e-100))
                self.run_one_episode(e)
                return e - 100
            if not self.quiet:
                self.run_one_episode(total_steps)
                print('[Episode {}, Steps {}] - Mean score of {} episodes was {}'.format(e, steps, len(scores), mean_score))
                
            self.replay(self.batch_size)
        
        if not self.quiet: print('Did not solve after {} episodes'.format(e))
        return e

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()