# Inspired by https://keon.io/deep-q-learning/
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
import copy


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
    def __init__(self, n_episodes=2000, min_reward=-150, max_env_steps=10000, gamma=0.99, exploration_steps=10_000, batch_size=128, quiet=False):
        self.memory = deque(maxlen=1000000)
        self.env = gym.make('Pendulum-v0')
        self.gamma = gamma
        self.critic_alpha = 0.01
        self.actor_alpha = 0.001
        self.critic_alpha_decay = 0.0
        self.actor_alpha_decay = 0.0001
        self.n_episodes = n_episodes
        self.min_reward = min_reward
        self.max_episode_steps = max_env_steps
        self.exploration_steps = exploration_steps
        self.batch_size = batch_size
        self.tau = 0.1
        self.quiet = quiet

        # Init models
        self.init_models()

    def output_scale(self, x):
        return x*2
    
    def output_scale_deriv(self, z):
        return np.ones(z.shape)*2
        
    def init_models(self):
        self.actor_fully_connected = fcn.FullyConnected(sizes=[3, 128, 64, 1], activations=["linear", "relu", "relu", "tanh"], scale_method=None, optimizer="nadam", lr=self.actor_alpha, lr_decay=self.actor_alpha_decay)
        self.actor_fully_connected.custom_output_scaler(custom_func=(self.output_scale), custom_func_deriv=(self.output_scale_deriv))
        self.actor_model = neuralnetwork.NeuralNetwork([self.actor_fully_connected], loss_fn="mean_squared_error")

        self.target_actor_fully_connected = fcn.FullyConnected(sizes=[3, 128, 64, 1], activations=["linear", "relu", "relu", "tanh"], scale_method=None, optimizer="nadam", lr=self.actor_alpha, lr_decay=self.actor_alpha_decay)
        self.target_actor_fully_connected.custom_output_scaler(custom_func=(self.output_scale), custom_func_deriv=(self.output_scale_deriv))
        self.target_actor_model = neuralnetwork.NeuralNetwork([self.target_actor_fully_connected], loss_fn="mean_squared_error")
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_actor_model.set_biases(self.actor_model.get_biases())

        self.critic_fully_connected = fcn.FullyConnected(sizes=[4, 24, 48, 1], activations=["linear", "relu", "relu", "linear"], scale_method=None, optimizer="nadam", lr=self.critic_alpha, lr_decay=self.critic_alpha_decay)
        self.critic_model = neuralnetwork.NeuralNetwork([self.critic_fully_connected], loss_fn="mean_squared_error")

        self.target_critic_fully_connected = fcn.FullyConnected(sizes=[4, 24, 48, 1], activations=["linear", "relu", "relu", "linear"], scale_method=None, optimizer="nadam", lr=self.critic_alpha, lr_decay=self.critic_alpha_decay)
        self.target_critic_model = neuralnetwork.NeuralNetwork([self.target_critic_fully_connected], loss_fn="mean_squared_error")
        self.target_critic_model.set_weights(self.critic_model.get_weights())
        self.target_critic_model.set_biases(self.critic_model.get_biases())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, step, noise_scale=0.1):
        if (step <= self.exploration_steps):
            action = self.env.action_space.sample()
        else:
            action = [self.actor_model.feedforward(state)]
        action += noise_scale * np.random.randn(1)
        return action
    
    def preprocess_state(self, state):
        return np.reshape(state, [1, 1, 3])
    
    def preprocess_critic_input(self, state, action):
        return np.concatenate((np.array(state).reshape((-1, 1, 3)), np.array(action).reshape((-1,1,1))), axis=2)


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        
        batch_states = np.array([i[0] for i in minibatch]).reshape((len(minibatch), 1, 3))
        batch_actions = np.array([i[1] for i in minibatch]).reshape((len(minibatch), 1, 1))
        batch_rewards = np.array([i[2] for i in minibatch]).reshape((len(minibatch), 1, 1))
        batch_next_states = np.array([i[3] for i in minibatch]).reshape((len(minibatch), 1, 3))
        batch_dones = np.array([i[4] for i in minibatch]).reshape((len(minibatch), 1, 1))
        batch_inputs = self.preprocess_critic_input(batch_states,batch_actions)
        y_target = (batch_rewards + (np.invert(batch_dones) * (self.gamma * self.target_critic_model.feedforward(self.preprocess_critic_input(batch_next_states, self.target_actor_model.feedforward(batch_next_states)))))).reshape((len(minibatch), 1, 1))
        self.critic_model.sgd_fit(batch_inputs, y_target, batch_size=batch_size, epochs=1, train_pct=1.0, shuffle_inputs=False, random_states=None)
        
        critic_grads = self.critic_model.gradients(self.preprocess_critic_input(batch_states, self.actor_model.feedforward(batch_states)), 0, "zLayer", grad_ys=None)[0][0][0][:,3]
        self.actor_model.sgd_fit(batch_states,None,grad_ys=-critic_grads,batch_size=batch_size, epochs=1, train_pct=1.0, shuffle_inputs=False, random_states=None)
        self.target_update()

    def target_update(self):
        self.target_critic_model.set_weights((self.tau * np.asarray(self.critic_model.get_weights()) + (1 - self.tau) * np.asarray(self.target_critic_model.get_weights())).tolist())
        self.target_critic_model.set_biases((self.tau * np.asarray(self.critic_model.get_biases()) + (1 - self.tau) * np.asarray(self.target_critic_model.get_biases())).tolist())
        
        self.target_actor_model.set_weights((self.tau * np.asarray(self.actor_model.get_weights()) + (1 - self.tau) * np.asarray(self.target_actor_model.get_weights())).tolist())
        self.target_actor_model.set_biases((self.tau * np.asarray(self.actor_model.get_biases()) + (1 - self.tau) * np.asarray(self.target_actor_model.get_biases())).tolist())
        
    def run_one_episode(self, total_steps):
        state = self.preprocess_state(self.env.reset())
        done = False
        rewards = 0
        for _ in range(100):
            self.env.render()
            action = self.choose_action(state, total_steps, 0)
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
            for _ in range(self.max_episode_steps):
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
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e-100))
                self.run_one_episode(e)
                return e - 100
            if e % 100 == 0 and not self.quiet:
                self.run_one_episode(total_steps)
                print('[Episode {}, Steps {}] - Mean score of 100 episodes was {}'.format(e, steps, mean_score))
                
            self.replay(self.batch_size)
        
        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()