import pdb
import random
import numpy as np
import math

import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_layers, dim, dropout_rate):
        super(DQN, self).__init__()
        self.in_layer = nn.Linear(n_observations, dim)
        self.layers = nn.Sequential(
            *[nn.Linear(dim,dim) for _ in range(n_layers)]
        )
        self.out_layer = nn.Linear(dim, n_actions)
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout with 20% probability

    def forward(self, x):
        x = F.normalize(x,dim=0)
        x = F.tanh(self.in_layer(x))
        for layer in self.layers:
            x = F.tanh(layer(x))
            x = self.dropout(x)
        x = self.out_layer(x)
        return x

class simulator:
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer

    ### VARY THESE TO SEE HOW PERFORMANCE IMPROVES ###

    def __init__(self, states, actions):
        self.BATCH_SIZE = 32
        self.GAMMA = 0.89
        self.EPS_START = 1.0
        self.EPS_END = 0.05
        # self.EPS_DECAY = 150000 <- opted to define it according to simulation length
        self.TAU = 0.04
        self.LR = 4e-5
        self.dropout_rate = 0.2
        self.n_layers = 3
        self.dim = 256

        # Store action space
        self.actions = actions
        # Store state space
        self.states = states
        self.policy_net = DQN(len(states), len(actions), self.n_layers, self.dim, self.dropout_rate).to(device)
        self.target_net = DQN(len(states), len(actions), self.n_layers, self.dim, self.dropout_rate).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.episode = 0
        self.eps_thresh = 0
        self.loss=0


    def select_action(self, state, decay):
        global steps_done
        sample = random.random()
        self.eps_thresh = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.episode / decay)
        if sample > self.eps_thresh:
            with torch.no_grad():
                # t.max(0) will return the largest value. Index indicates which action
                # returns highest reward.
                a = self.policy_net(state.to(device)).max(0).indices.view(1)
                return torch.tensor([a],device=device,dtype=torch.long)
        else:
            return torch.tensor([random.randint(0,len(self.actions)-1)], device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch
        batch = Transition(*zip(*transitions))
        del transitions

        # Gather next states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        next_states = torch.stack([s for s in batch.next_state if s is not None]).to(device)
        
        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.cat(batch.action).unsqueeze(0).to(device)
        reward_batch = torch.cat(batch.reward).unsqueeze(0).to(device)

        # Compute Q function values (taken from NN)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for next_states are computed based on the "older"
        # target_net; selecting their best reward with max(1).values
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(next_states).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.loss = loss.item()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()