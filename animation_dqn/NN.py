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
    
def select_action_trained(network, state_tensor, actions_list):
    """
    Selects the best action using a trained network for inference.
    No exploration, always chooses the action with highest Q-value.
    
    Args:
        network: The loaded trained network (e.g., model["network"])
        state_tensor: Current state as a tensor
        actions_list: List of possible actions (e.g., model["actions"])
    
    Returns:
        The chosen action tuple (e.g., (200, 0))
    """
    with torch.no_grad():
        # Get Q-values for all actions
        q_values = network(state_tensor)
        # Find the index of the action with the highest Q-value
        action_index = torch.argmax(q_values).item()
    
    # Return the actual action tuple from the list
    return actions_list[action_index]

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_layers, dim, dropout_rate, pretrained_weights_path=None):
        super(DQN, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(n_observations, dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(dim, n_actions))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        if pretrained_weights_path:
            self.load_state_dict(torch.jit.load(pretrained_weights_path, map_location=device).state_dict())
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class simulator:
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer

    ### VARY THESE TO SEE HOW PERFORMANCE IMPROVES ###

    def __init__(self, states, actions, pretrained_weights_path=None):
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 1.0
        self.EPS_END = 0.01
        self.TAU = 0.005
        self.LR = 1e-4
        self.dropout_rate = 0.1
        self.n_layers = 3
        self.dim = 256

        # Store action space
        self.actions = actions
        # Store state space
        self.states = states
        self.policy_net = DQN(len(states), len(actions), self.n_layers, self.dim, self.dropout_rate, pretrained_weights_path).to(device)
        self.target_net = DQN(len(states), len(actions), self.n_layers, self.dim, self.dropout_rate, pretrained_weights_path).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.episode = 0
        self.eps_thresh = 0
        self.loss=0


    def select_action(self, state, decay):
        # Calculate epsilon threshold
        self.eps_thresh = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.episode / decay)
        
        if random.random() > self.eps_thresh:
            with torch.no_grad():
                # Use the policy network to select the best action
                q_values = self.policy_net(state)        # Shape: [num_actions]
                action_index = q_values.argmax()         # Get the index of the max Q-value
                return action_index.unsqueeze(0)         # Return shape [1] to keep it as a tensor
        else:
            # Explore: choose a random action index
            action_index = random.randint(0, len(self.actions) - 1)
            return torch.tensor([action_index], device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Compute masks and stack tensors
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                    device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        
        # Move everything to device
        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        non_final_next_states = non_final_next_states.to(device)
        
        # Compute Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next state values
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch.squeeze()
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.loss = loss.item()