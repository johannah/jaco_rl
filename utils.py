import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class BasePolicy():
    def __init__(
        self,
        state_dim,
        action_dim,
        min_action,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        seed=1222,
        device='cpu'
    ):
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.max_action = max_action
        self.min_action = min_action
        self.action_dim = action_dim
        self.state_dim = state_dim

    def select_action(self, state):
        action = np.random.uniform(low=self.min_action, high=self.max_action, size=self.action_dim)
        return action

    def train(self, replay_buffer, batch_size=100):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass


class RandomPolicy(BasePolicy):
    def select_action(self, state):
        action = np.random.uniform(low=self.min_action, high=self.max_action, size=self.action_dim)
        return action
