import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=512):  # Increased hidden size
        super(ActorCritic, self).__init__()
        
        # Enhanced shared layers with more capacity
        self.shared = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),  # Additional layer
            nn.ReLU(),
        )
        
        # Actor head with more capacity
        self.actor = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, num_actions)
        )
        
        # Critic head with more capacity  
        self.critic = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )
        
        # Learnable log std with smaller initialization
        self.log_std = nn.Parameter(torch.zeros(1, num_actions) * 0.5)
        
    def forward(self, x):
        shared_out = self.shared(x)
        value = self.critic(shared_out)
        mean = torch.tanh(self.actor(shared_out))
        std = self.log_std.exp().expand_as(mean)
        
        return mean, std, value

class PPOAgent:
    def __init__(self, num_inputs, num_actions, pretrained_weights, **hyperparams):
        self.model = ActorCritic(num_inputs, num_actions, 
                               hidden_size=hyperparams.get('dim', 256)).to(device)
        
        # PPO hyperparameters
        self.ppo_epochs = hyperparams.get('ppo_epochs', 4)
        self.mini_batch_size = hyperparams.get('batch_size', 64)
        self.clip_param = hyperparams.get('clip', 0.2)
        self.gamma = hyperparams.get('gamma', 0.99)
        self.gae_lambda = hyperparams.get('gae_lambda', 0.95)
        self.lr = hyperparams.get('lr', 3e-4)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Memory buffers - we'll store entire episodes
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_rewards = []
        self.episode_dones = []
        
        # Load pretrained weights if provided
        if pretrained_weights and pretrained_weights[0]:
            self.model.load_state_dict(torch.load(pretrained_weights[0]))
    
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std, value = self.model(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
        return action.cpu().numpy()[0], log_prob.item(), value.item()
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob)
        self.episode_values.append(value)
        self.episode_rewards.append(reward)
        self.episode_dones.append(done)
    
    def compute_gae(self, next_value):
        values = self.episode_values + [next_value]
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(self.episode_rewards))):
            delta = self.episode_rewards[step] + self.gamma * values[step + 1] * (1 - self.episode_dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.episode_dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return returns, advantages
    
    def learn(self):
        if len(self.episode_states) < self.mini_batch_size:
            return
            
        # Estimate next value for the last state
        if self.episode_dones[-1]:
            next_value = 0
        else:
            with torch.no_grad():
                next_state = torch.FloatTensor(self.episode_states[-1]).unsqueeze(0).to(device)
                _, _, next_value_tensor = self.model(next_state)
                next_value = next_value_tensor.item()
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states)).to(device)
        actions = torch.FloatTensor(np.array(self.episode_actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.episode_log_probs)).to(device)
        returns = torch.FloatTensor(np.array(returns)).to(device)
        advantages = torch.FloatTensor(np.array(advantages)).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO optimization
        for _ in range(self.ppo_epochs):
            # Generate random indices
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # Process in mini-batches
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get current policy
                mean, std, values = self.model(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().mean()
                
                # Policy loss
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.02 * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
        
        # Clear episode memory
        self.clear_memory()
    
    def clear_memory(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_rewards = []
        self.episode_dones = []

def select_action_trained(model, state_tensor):
    """Select action using trained model for inference"""
    with torch.no_grad():
        mean, std, _ = model(state_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
        return action.cpu().numpy()[0]