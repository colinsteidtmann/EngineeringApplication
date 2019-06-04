# Inspired by https://keon.io/deep-q-learning/
# Adapted from https://gist.github.com/n1try/2a6722407117e4d668921fce53845432
import sys
sys.path.append("..")
import random
import gym
import math
import numpy as np
from collections import deque
from NeuralNets import ConvNet as cnn
from NeuralNets import FullyConnected as fcn
from NeuralNets import NeuralNetwork as neuralnetwork



class DQNCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=100, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v0')
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.fully_connected = fcn.FullyConnected(sizes=[4, 24, 48, 2], activations=["linear","relu", "relu", "linear"], scale_method=None, optimizer="nadam", lr=.01, lr_decay=(0.0))
        self.model = neuralnetwork.NeuralNetwork([self.fully_connected], loss_fn="mean_squared_error")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.feedforward(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 1, 4])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        
        batch_states = np.array([i[0] for i in minibatch]).reshape((len(minibatch), 1, 4))
        batch_actions = np.array([i[1] for i in minibatch]).reshape((len(minibatch),1,1))
        batch_rewards = np.array([i[2] for i in minibatch]).reshape((len(minibatch), 1, 1))
        batch_next_states = np.array([i[3] for i in minibatch]).reshape((len(minibatch), 1, 4))
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
            self.env.render()
            action = self.choose_action(state, total_steps, test=True)
            next_state, reward, done, _ = self.env.step(action)
            next_state = self.preprocess_state(next_state)
            state = next_state
            rewards += reward
        return rewards
    

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                self.run_one_episode(e)
                return e - 100
            if e % 100 == 0 and not self.quiet:
                self.run_one_episode(e)
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

            self.replay(self.batch_size)
        
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()