"""
MADDPG Agent Implementation
Handles training, action selection, and network updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .actor_network import ActorNetwork
from .critic_network import CriticNetwork
from training.replay_buffer import ReplayBuffer


class MADDPGAgent:
    """Multi-Agent DDPG Agent for UAV Offloading"""
    
    def __init__(
        self,
        state_dim=537,
        offload_dim=5,
        continuous_dim=6,
        action_dim=11,
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Actor Networks
        self.actor = ActorNetwork(state_dim, offload_dim, continuous_dim).to(device)
        self.actor_target = ActorNetwork(state_dim, offload_dim, continuous_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic Networks
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Exploration noise
        self.exploration_noise = 0.1
        self.noise_decay = 0.995
        self.min_noise = 0.01
    
    def select_action(self, state, explore=True):
        """
        Select action from current policy
        
        Args:
            state: np.array of shape (537,)
            explore: bool, whether to add exploration noise
        
        Returns:
            action: np.array of shape (11,)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            offload_logits, continuous_action = self.actor(state_tensor)
            
            # Sample discrete action (offload decision)
            if explore:
                offload_probs = torch.softmax(offload_logits, dim=-1)
                offload_idx = torch.multinomial(offload_probs, 1).item()
            else:
                offload_idx = torch.argmax(offload_logits, dim=-1).item()
            
            # One-hot encode offload decision
            offload_onehot = np.zeros(5)
            offload_onehot[offload_idx] = 1.0
            
            # Get continuous action
            continuous_np = continuous_action.cpu().numpy().flatten()
            
            # Add exploration noise
            if explore:
                noise = np.random.normal(0, self.exploration_noise, size=continuous_np.shape)
                continuous_np = np.clip(continuous_np + noise, -1, 1)
        
        self.actor.train()
        
        # Combine: [offload_onehot(5) + continuous(6)] = 11
        action = np.concatenate([offload_onehot, continuous_np])
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self, batch_size=64):
        """
        Perform one training step
        
        Returns:
            actor_loss: float
            critic_loss: float
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ========== Update Critic ==========
        with torch.no_grad():
            # Get next actions from target actor
            next_offload_logits, next_continuous = self.actor_target(next_states)
            next_offload = torch.softmax(next_offload_logits, dim=-1)
            next_actions = torch.cat([next_offload, next_continuous], dim=-1)
            
            # Compute target Q-value
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-value
        current_q = self.critic(states, actions)
        
        # Critic loss (MSE)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ========== Update Actor ==========
        offload_logits, continuous = self.actor(states)
        offload_probs = torch.softmax(offload_logits, dim=-1)
        pred_actions = torch.cat([offload_probs, continuous], dim=-1)
        
        # Actor loss (maximize Q-value)
        actor_loss = -self.critic(states, pred_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ========== Soft Update Target Networks ==========
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        # Decay exploration noise
        self.exploration_noise = max(
            self.min_noise,
            self.exploration_noise * self.noise_decay
        )
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, source, target):
        """Polyak averaging for target network update"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save_checkpoint(self, filepath):
        """Save agent checkpoint"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'exploration_noise': self.exploration_noise
        }, filepath)
        print(f"💾 Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.exploration_noise = checkpoint['exploration_noise']
        print(f"✅ Checkpoint loaded: {filepath}")
