import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, states):
        x = torch.tanh(self.fc1(states))
        x = torch.tanh(self.fc2(x))
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        return mu, std


class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(SACActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        self.log_std_min = -20
        self.log_std_max = 2
        self.eps = 1e-6
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.log_std.weight)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, states):
        mus, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(mus, stds)
        xs = normals.rsample()
        actions = torch.tanh(xs)
        log_probs = normals.log_prob(
            xs) - torch.log(1-actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(mus)


class QCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(QCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.q.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q


class TwinnedQCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(TwinnedQCritic, self).__init__()
        self.Q1 = QCritic(state_dim, action_dim, hidden_size)
        self.Q2 = QCritic(state_dim, action_dim, hidden_size)

    def reset_parameters(self):
        self.Q1.reset_parameters()
        self.Q2.reset_parameters()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


class VCritic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(VCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.v(x)
        return v
