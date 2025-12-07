"""
maddpg_agent.py
Multi-Agent Deep Deterministic Policy Gradient Agent با Replay Buffer
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List
from collections import deque
import random


class ReplayBuffer:
    """
    Replay Buffer برای ذخیره و نمونه‌برداری تجربیات
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, 
             global_state, global_next_state, global_actions):
        """ذخیره یک تجربه"""
        self.buffer.append((
            state, action, reward, next_state, done,
            global_state, global_next_state, global_actions
        ))
    
    def sample(self, batch_size: int):
        """نمونه‌برداری تصادفی"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones, \
        global_states, global_next_states, global_actions = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(global_states),
            np.array(global_next_states),
            np.array(global_actions)
        )
    
    def __len__(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """
    Actor Network: state -> action
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        """Forward pass"""
        return self.network(state)


class CriticNetwork(nn.Module):
    """
    Critic Network: (global_state, global_actions) -> Q-value
    """
    def __init__(self, total_state_dim: int, total_action_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, global_state, global_actions):
        """Forward pass"""
        x = torch.cat([global_state, global_actions], dim=1)
        return self.network(x)


class MADDPGAgent:
    """
    MADDPG Agent با Centralized Critic و Decentralized Actor
    """
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        total_state_dim: int,
        total_action_dim: int,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        buffer_size: int = 50000,
        batch_size: int = 256,
        device: str = 'cpu'
    ):
        """
        Args:
            agent_id: شناسه عامل
            state_dim: بعد فضای مشاهده این عامل
            action_dim: بعد فضای اکشن این عامل
            total_state_dim: بعد کل فضای مشاهدات (تمام عاملان)
            total_action_dim: بعد کل فضای اکشن‌ها (تمام عاملان)
            lr_actor: learning rate برای Actor
            lr_critic: learning rate برای Critic
            gamma: ضریب تخفیف
            tau: ضریب soft update
            buffer_size: اندازه Replay Buffer
            batch_size: اندازه batch برای آموزش
            device: 'cpu' یا 'cuda'
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.total_state_dim = total_state_dim
        self.total_action_dim = total_action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = CriticNetwork(total_state_dim, total_action_dim).to(self.device)
        self.critic_target = CriticNetwork(total_state_dim, total_action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """
        انتخاب اکشن با استفاده از Actor Network
        
        Args:
            state: مشاهده فعلی
            add_noise: اضافه کردن نویز برای Exploration
        
        Returns:
            action: اکشن انتخاب شده (scaled به [0, 1])
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        # Scale از [-1, 1] به [0, 1]
        action = (action + 1) / 2
        
        # Add exploration noise
        if add_noise:
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action + noise, 0, 1)
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        global_state: np.ndarray,
        global_next_state: np.ndarray,
        global_actions: np.ndarray
    ):
        """
        ذخیره تجربه در Replay Buffer
        """
        self.replay_buffer.push(
            state, action, reward, next_state, done,
            global_state, global_next_state, global_actions
        )
    
    def update(self, all_agents: List['MADDPGAgent']) -> Tuple[float, float]:
        """
        ✅ آموزش Actor و Critic Networks با استفاده از MADDPG
        
        Args:
            all_agents: لیست تمام عاملان MADDPG (برای دسترسی به actor_target آن‌ها)
        
        Returns:
            (critic_loss, actor_loss): مقادیر Loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, \
        global_states, global_next_states, global_actions = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        global_states = torch.FloatTensor(global_states).to(self.device)
        global_next_states = torch.FloatTensor(global_next_states).to(self.device)
        global_actions = torch.FloatTensor(global_actions).to(self.device)
        
        # تعداد عاملان را محاسبه کن
        n_agents = len(all_agents)
        
        # ==================== Update Critic ====================
        with torch.no_grad():
            # محاسبه next_actions برای همه عاملان از طریق target actors
            next_actions_list = []
            
            for i, agent in enumerate(all_agents):
                # محاسبه indices برای state هر عامل
                start_idx = i * agent.state_dim
                end_idx = start_idx + agent.state_dim
                
                # استخراج state این عامل از global_next_states
                agent_next_state = global_next_states[:, start_idx:end_idx]
                
                # استفاده از actor_target آن عامل
                agent_next_action = agent.actor_target(agent_next_state)
                
                # Scale اکشن از [-1, 1] به [0, 1] (فضای environment)
                agent_next_action = (agent_next_action + 1) / 2
                next_actions_list.append(agent_next_action)
            
            # Concatenate تمام next actions
            next_actions_all = torch.cat(next_actions_list, dim=1)
            
            # محاسبه Target Q-value
            target_q = self.critic_target(global_next_states, next_actions_all)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # محاسبه Current Q-value
        current_q = self.critic(global_states, global_actions)
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ==================== Update Actor ====================
        # محاسبه predicted actions برای همه عاملان
        predicted_actions_list = []
        
        for i, agent in enumerate(all_agents):
            # محاسبه indices برای state هر عامل
            start_idx = i * agent.state_dim
            end_idx = start_idx + agent.state_dim
            
            # استخراج state این عامل
            agent_state = global_states[:, start_idx:end_idx]
            
            if i == self.agent_id:
                # برای این عامل، از actor جدید استفاده کن
                agent_action = self.actor(agent_state)
            else:
                # برای عاملان دیگر، از actor آن‌ها استفاده کن (detached)
                agent_action = agent.actor(agent_state).detach()
            
            # Scale به [0, 1]
            agent_action = (agent_action + 1) / 2
            predicted_actions_list.append(agent_action)
        
        # Concatenate تمام predicted actions
        predicted_actions_all = torch.cat(predicted_actions_list, dim=1)
        
        # Actor loss (هدف: بیشینه‌سازی Q-value)
        actor_loss = -self.critic(global_states, predicted_actions_all).mean()
        
        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft Update Target Networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        Soft update: target = tau * source + (1 - tau) * target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, filepath: str):
        """ذخیره وزن‌های شبکه"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """بارگذاری وزن‌های شبکه"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
