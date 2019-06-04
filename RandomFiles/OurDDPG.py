import numpy as np
import utils


# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x)) 
		return x 


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, x, u):
		x = F.relu(self.l1(torch.cat([x, u], 1)))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x 


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action):
        self.actor = neuralnetwork.NeuralNetwork(sizes=[3, 400, 300, 1],activations=["linear", "relu", "relu", "tanh"], optimizer="nadam", lr=self.actor_alpha, lr_decay=self.actor_alpha_decay)
        self.actor.custom_output_scaler(custom_func=(self.output_scale),custom_func_deriv=(self.output_scale_deriv))
        self.actor_target = neuralnetwork.NeuralNetwork(sizes=[3, 400, 300, 1], activations=["linear", "relu", "relu", "tanh"], optimizer="nadam", lr=self.actor_alpha, lr_decay=self.actor_alpha_decay)
        self.actor_target.custom_output_scaler(custom_func=(self.output_scale),custom_func_deriv=(self.output_scale_deriv))
        self.actor_target.set_weights(self.actor.get_weights())
        self.actor_target.set_biases(self.actor.get_biases())

        self.critic = neuralnetwork.NeuralNetwork(sizes=[4, 400, 300, 1],activations=["linear", "relu", "relu", "linear"], optimizer="nadam", lr=self.critic_alpha, lr_decay=self.critic_alpha_decay)
        self.critic_target = neuralnetwork.NeuralNetwork(sizes=[4, 400, 300, 1],activations=["linear", "relu", "relu", "linear"], optimizer="nadam", lr=self.critic_alpha, lr_decay=self.critic_alpha_decay)
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_target.set_biases(self.critic.get_biases())	


	def select_action(self, state):
		return self.actor.feedforward(state)


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

		for it in range(iterations):

			# Sample replay buffer 
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))