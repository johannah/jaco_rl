import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_policy_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_policy_action = max_policy_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_policy_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)


    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_policy_action=1.0, discount=0.99, tau=0.005, device='cpu', **kwargs):
        self.device = device
        self.actor = Actor(state_dim, action_dim, max_policy_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.discount = discount
        self.tau = tau
        self.step = 0
        self.loss_dict = {'actor':[], 'critic':[], 'step':[]}

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, step, replay_buffer, batch_size=100):
        self.step = step
        # Sample replay buffer

        state, action, reward, next_state, not_done, _, _ = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        not_done = torch.FloatTensor(not_done).to(self.device)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

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
        self.loss_dict = {'actor':[], 'critic':[], 'step':[]}

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        model_dict =  {'critic':self.critic.state_dict(), 
                      'actor':self.actor.state_dict(), 
                      'critic_optimizer':self.critic_optimizer.state_dict(), 
                      'actor_optimizer':self.actor_optimizer.state_dict(),
                      'loss_dict':self.loss_dict, 
                      }
        torch.save(model_dict, filepath)

    def get_loss_plot_data(self):
        plot_dict =  {'critic':(self.loss_dict['step'], self.loss_dict['critic']), 
                'actor':(self.loss_dict['step'], self.loss_dict['actor'])}
        return plot_dict

    def load(self, filepath):
        print("OurDDPG loading {}".format(filepath))
        model_dict = torch.load(filepath)
        self.critic.load_state_dict(model_dict['critic'])
        self.actor.load_state_dict(model_dict['actor'])
        self.critic_optimizer.load_state_dict(model_dict['critic_optimizer'])
        self.actor_optimizer.load_state_dict(model_dict['actor_optimizer'])
        self.loss_dict = model_dict['loss_dict']
        # TODO this is wrong
        if len(self.loss_dict['step']):
            self.step = self.loss_dict['step'][-1]
        else:
            self.step = 0

