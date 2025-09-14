import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, n_layers, dim, dropout_rate):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        
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
        
        # Output layers for mean and log_std
        self.network = nn.Sequential(*layers)
        self.mu = nn.Linear(dim, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.network(x)
        mu = torch.tanh(self.mu(x))  # Output in [-1, 1] range
        # Handle different input shapes
        if x.dim() == 1:  # Single sample
            std = torch.exp(self.log_std)
        # Batch of samples
        else:
            std = torch.exp(self.log_std).unsqueeze(0).expand(x.size(0), -1)
        return mu, std

class CriticNetwork(nn.Module):
    def __init__(self, n_observations, n_layers, dim, dropout_rate):
        super(CriticNetwork, self).__init__()
        
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
        layers.append(nn.Linear(dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class PPOAgent:
    def __init__(self, n_observations, n_actions, **hyperparams):
        # Hyperparameters
        self.gamma = hyperparams.get('gamma', 0.99)
        self.gae_lambda = hyperparams.get('gae_lambda', 0.95)
        self.ppo_epochs = hyperparams.get('ppo_epochs', 10)
        self.batch_size = hyperparams.get('batch_size', 64)
        self.clip = hyperparams.get('clip', 0.2)
        self.lr = hyperparams.get('lr', 3e-4)
        self.n_layers = hyperparams.get('n_layers', 3)
        self.dim = hyperparams.get('dim', 256)
        self.dropout_rate = hyperparams.get('dropout_rate', 0.1)
        
        self.n_actions = n_actions
        self.actor = ActorNetwork(n_observations, n_actions, self.n_layers, self.dim, self.dropout_rate).to(device)
        self.critic = CriticNetwork(n_observations, self.n_layers, self.dim, self.dropout_rate).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.memory = PPOMemory(self.batch_size)
        
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        with torch.no_grad():
            mu, std = self.actor(state)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            action = torch.clamp(action, -1.0, 1.0)  # Ensure action is in [-1, 1]
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state)
            
            return action.cpu().numpy(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, prob, val, reward, done):
        self.memory.store_memory(state, action, prob, val, reward, done)
    
    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return
            
        # Convert to tensors
        states = torch.tensor(np.array(self.memory.states), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(self.memory.actions), dtype=torch.float).to(device)
        old_probs = torch.tensor(self.memory.probs, dtype=torch.float).to(device)
        old_vals = torch.tensor(self.memory.vals, dtype=torch.float).to(device)
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float).to(device)
        dones = torch.tensor(self.memory.dones, dtype=torch.float).to(device)
        
        # Calculate advantages and returns using GAE
        advantages = torch.zeros_like(rewards).to(device)
        returns = torch.zeros_like(rewards).to(device)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = old_vals[t + 1] * (1 - dones[t])
            
            delta = rewards[t] + self.gamma * next_value - old_vals[t]
            if t < len(rewards) - 1:
                advantages[t] = delta + self.gamma * self.gae_lambda * advantages[t + 1] * (1 - dones[t])
            else:
                advantages[t] = delta
        
        returns = advantages + old_vals
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            batches = self.memory.generate_batches()
            
            for batch in batches:
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_probs = old_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]
                
                # Critic loss
                values = self.critic(batch_states).squeeze()
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                critic_loss = F.mse_loss(values, batch_returns)
                
                # Actor loss
                mu, std = self.actor(batch_states)
                dist = torch.distributions.Normal(mu, std)
                new_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                prob_ratio = torch.exp(new_probs - batch_old_probs)
                weighted_probs = batch_advantages * prob_ratio
                weighted_clipped_probs = batch_advantages * torch.clamp(prob_ratio, 1 - self.clip, 1 + self.clip)
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Entropy bonus for exploration
                entropy = dist.entropy().mean()
                
                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                # Backpropagation
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        self.memory.clear_memory()

def select_action_trained(actor, state_tensor):
    """Select action using trained actor network for inference"""
    with torch.no_grad():
        mu, std = actor(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy()