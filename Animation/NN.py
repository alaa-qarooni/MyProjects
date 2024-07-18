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

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, n_actions)
        self.dropout = nn.Dropout(p=0.2)  # Dropout with 20% probability

    def forward(self, x):
        x = F.normalize(x,dim=0)
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        x = self.dropout(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
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
        self.BATCH_SIZE = 128
        self.GAMMA = 0.90
        self.EPS_START = 0.9
        self.EPS_END = 0.5
        # self.EPS_DECAY = 150000 <- opted to define it according to simulation length
        self.TAU = 0.05
        self.LR = 1e-5

        # Store action space
        self.actions = actions
        # Store state space
        self.states = states

        self.policy_net = DQN(len(states), len(actions)).to(device)
        self.target_net = DQN(len(states), len(actions)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        
        self.steps_done = 0


    def select_action(self, state, decay):
        global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / decay)
        self.steps_done += 1
        if sample > eps_threshold:
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
        next_states = torch.stack(batch.next_state).to(device)
        
        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.cat(batch.action).unsqueeze(0).to(device)
        reward_batch = torch.cat(batch.reward).unsqueeze(0).to(device)

        # Compute Q function values (taken from NN)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for next_states are computed based on the "older"
        # target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()