# models/actor_critic/maddpg_agent.py
"""
MADDPG Agent (Simplified Single-Agent DDPG)
با ابعاد state=114, action=7
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


# =====================================================================
# Replay Buffer
# =====================================================================
class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """ذخیره یک transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """نمونه‌برداری تصادفی"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# =====================================================================
# Actor Network
# =====================================================================
class Actor(nn.Module):
    """Deterministic Policy Network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.activation = nn.ReLU()
    
    def forward(self, state):
        """
        state: (B, state_dim)
        output: (B, action_dim) ∈ [-1, 1]
        """
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # [-1, 1]
        return action


# =====================================================================
# Critic Network
# =====================================================================
class Critic(nn.Module):
    """Q-Value Network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()
    
    def forward(self, state, action):
        """
        state: (B, state_dim)
        action: (B, action_dim)
        output: (B, 1) Q-value
        """
        x = torch.cat([state, action], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        q = self.q_out(x)
        return q


# =====================================================================
# MADDPG Agent
# =====================================================================
class MADDPGAgent:
    """
    MADDPG Agent (simplified to single-agent DDPG)
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=512,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.01,
        buffer_size=100000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    def select_action(self, state, noise=0.0):
        """
        انتخاب action با exploration noise
        
        Args:
            state: numpy array (state_dim,)
            noise: std of Gaussian noise
        
        Returns:
            action: numpy array (action_dim,)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        
        # Add noise for exploration
        if noise > 0:
            action += np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -1, 1)
        
        return action
    
    def update(self, batch_size=64):
        """
        به‌روزرسانی Actor و Critic
        
        Returns:
            actor_loss: float
            critic_loss: float
        """
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # Sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # -------------------- Update Critic --------------------
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # -------------------- Update Actor --------------------
        pred_action = self.actor(state)
        actor_loss = -self.critic(state, pred_action).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # -------------------- Soft Update Target Networks --------------------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, source, target):
        """Soft update: target = tau * source + (1-tau) * target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """ذخیره مدل"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load(self, filepath):
        """بارگذاری مدل"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        print(f"✅ Model loaded from: {filepath}")
