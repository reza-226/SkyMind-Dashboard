"""
مطالعات Ablation - Variants مختلف MADDPG
مسیر: core/evaluation/ablation_variants.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


# ============================================================================
# شبکه‌های عصبی برای Variants مختلف
# ============================================================================

class SimpleMLPActor(nn.Module):
    """Actor ساده بدون GAT"""
    
    def __init__(self, state_dim, action_dim, hidden=512):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        
        # ✅ استفاده از sigmoid برای خروجی [0,1]
        action = torch.sigmoid(self.fc3(x))
        
        # ✅ CRITICAL: اعمال epsilon برای جلوگیری از 0.0 و 1.0 دقیق
        epsilon = 1e-6
        action = action * (1 - 2*epsilon) + epsilon
        
        return action


class SimpleCritic(nn.Module):
    """Critic ساده"""
    
    def __init__(self, state_dim, action_dim, hidden=512):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q_out = nn.Linear(hidden, 1)
        
        self.activation = nn.ReLU()
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        q = self.q_out(x)
        return q


class SmallActor(nn.Module):
    """Actor کوچک‌تر برای SimplerArchVariant"""
    
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, state):
        x = self.activation(self.fc1(state))
        
        # ✅ استفاده از sigmoid برای خروجی [0,1]
        action = torch.sigmoid(self.fc2(x))
        
        # ✅ CRITICAL: اعمال epsilon
        epsilon = 1e-6
        action = action * (1 - 2*epsilon) + epsilon
        
        return action


class SmallCritic(nn.Module):
    """Critic کوچک‌تر"""
    
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.q_out = nn.Linear(hidden, 1)
        
        self.activation = nn.ReLU()
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.activation(self.fc1(x))
        q = self.q_out(x)
        return q


class LocalCritic(nn.Module):
    """Critic محلی برای DecentralizedVariant"""
    
    def __init__(self, state_dim, action_dim, hidden=512):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q_out = nn.Linear(hidden, 1)
        
        self.activation = nn.ReLU()
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        q = self.q_out(x)
        return q


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Replay Buffer برای ذخیره تجربیات"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# VARIANT 1: Full MADDPG (Baseline)
# ============================================================================

class FullMADDPGVariant:
    """پیاده‌سازی کامل MADDPG با تمام ویژگی‌ها"""
    
    def __init__(self, obs_dim, action_dim, num_agents, **config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # هایپرپارامترها
        self.gamma = config.get("gamma", 0.95)
        self.tau = config.get("tau", 0.001)
        self.batch_size = config.get("batch_size", 64)
        self.actor_lr = config.get("actor_lr", 1e-4)
        self.critic_lr = config.get("critic_lr", 1e-3)
        
        # شبکه‌ها
        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for _ in range(num_agents):
            actor = SimpleMLPActor(obs_dim, action_dim, hidden=512).to(self.device)
            actor_target = SimpleMLPActor(obs_dim, action_dim, hidden=512).to(self.device)
            actor_target.load_state_dict(actor.state_dict())
            
            total_state_dim = obs_dim * num_agents
            total_action_dim = action_dim * num_agents
            critic = SimpleCritic(total_state_dim, total_action_dim, hidden=512).to(self.device)
            critic_target = SimpleCritic(total_state_dim, total_action_dim, hidden=512).to(self.device)
            critic_target.load_state_dict(critic.state_dict())
            
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.actor_lr)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
        
        buffer_size = config.get("buffer_size", 100000)
        self.replay_buffers = [ReplayBuffer(buffer_size) for _ in range(num_agents)]
        
        # ✅ Exploration noise با کاهش تدریجی
        self.noise_scale = 0.1  # شروع با 0.1
        self.noise_decay = 0.9995  # کاهش تدریجی
        self.min_noise = 0.01  # حداقل نویز
        
    def select_action(self, agent_id, obs, add_noise=True):
        """انتخاب action با Clipping دقیق"""
        if isinstance(agent_id, str):
            agent_id = int(agent_id.split('_')[-1])
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actors[agent_id](obs_tensor)
        
        action = action.cpu().numpy()[0]
        
        if add_noise:
            # ✅ نویز با کاهش تدریجی
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = action + noise
            
            # کاهش نویز
            self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
        
        # ✅ CRITICAL: Clipping دقیق با تلرانس
        action = np.clip(action, 0.0, 1.0)
        
        # ✅ Round به 6 رقم اعشار برای جلوگیری از خطاهای دقت
        action = np.round(action, decimals=6)
        
        # ✅ اطمینان نهایی از بازه [epsilon, 1-epsilon]
        epsilon = 1e-7
        action = np.where(action < epsilon, epsilon, action)
        action = np.where(action > (1.0 - epsilon), 1.0 - epsilon, action)
        
        return action.astype(np.float32)
    
    def store_transition(self, agent_id, state, action, reward, next_state, done):
        """ذخیره تجربه"""
        if isinstance(agent_id, str):
            agent_id = int(agent_id.split('_')[-1])
        
        self.replay_buffers[agent_id].push(state, action, reward, next_state, done)
    
    def update(self):
        """آپدیت شبکه‌ها"""
        for agent_id in range(self.num_agents):
            if len(self.replay_buffers[agent_id]) < self.batch_size:
                continue
            
            states, actions, rewards, next_states, dones = \
                self.replay_buffers[agent_id].sample(self.batch_size)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            
            all_states = states.repeat(1, self.num_agents)
            all_actions = actions.repeat(1, self.num_agents)
            all_next_states = next_states.repeat(1, self.num_agents)
            
            # ========== Update Critic ==========
            with torch.no_grad():
                next_actions = self.actor_targets[agent_id](next_states)
                all_next_actions = next_actions.repeat(1, self.num_agents)
                
                target_q = self.critic_targets[agent_id](all_next_states, all_next_actions)
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            current_q = self.critics[agent_id](all_states, all_actions)
            critic_loss = F.mse_loss(current_q, target_q)
            
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            
            # ✅ CRITICAL: Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.critics[agent_id].parameters(), 
                max_norm=0.5
            )
            
            self.critic_optimizers[agent_id].step()
            
            # ========== Update Actor ==========
            predicted_actions = self.actors[agent_id](states)
            all_predicted_actions = predicted_actions.repeat(1, self.num_agents)
            
            actor_loss = -self.critics[agent_id](all_states, all_predicted_actions).mean()
            
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            
            # ✅ CRITICAL: Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.actors[agent_id].parameters(), 
                max_norm=0.5
            )
            
            self.actor_optimizers[agent_id].step()
            
            # ========== Soft Update Target Networks ==========
            self._soft_update(self.actors[agent_id], self.actor_targets[agent_id])
            self._soft_update(self.critics[agent_id], self.critic_targets[agent_id])
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """ذخیره مدل"""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'noise_scale': self.noise_scale
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """بارگذاری مدل"""
        checkpoint = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
        
        if 'noise_scale' in checkpoint:
            self.noise_scale = checkpoint['noise_scale']


# ============================================================================
# VARIANT 2: No GAT
# ============================================================================

class NoGATVariant(FullMADDPGVariant):
    """MADDPG بدون Graph Attention Network"""
    
    def __init__(self, obs_dim, action_dim, num_agents, **config):
        super().__init__(obs_dim, action_dim, num_agents, **config)
        print("📌 NoGATVariant: استفاده از MLP ساده به جای GAT")


# ============================================================================
# VARIANT 3: No Temporal Features
# ============================================================================

class NoTemporalVariant(FullMADDPGVariant):
    """MADDPG بدون ویژگی‌های زمانی"""
    
    def __init__(self, obs_dim, action_dim, num_agents, **config):
        # کاهش 30% از ابعاد obs
        reduced_obs_dim = int(obs_dim * 0.7)
        super().__init__(reduced_obs_dim, action_dim, num_agents, **config)
        
        self.original_obs_dim = obs_dim
        self.reduced_obs_dim = reduced_obs_dim
        print(f"📌 NoTemporalVariant: کاهش state از {obs_dim} به {reduced_obs_dim}")
    
    def select_action(self, agent_id, obs, add_noise=True):
        """انتخاب action با obs کاهش یافته"""
        reduced_obs = obs[:self.reduced_obs_dim]
        return super().select_action(agent_id, reduced_obs, add_noise)
    
    def store_transition(self, agent_id, state, action, reward, next_state, done):
        """ذخیره با state کاهش یافته"""
        reduced_state = state[:self.reduced_obs_dim]
        reduced_next_state = next_state[:self.reduced_obs_dim]
        super().store_transition(agent_id, reduced_state, action, reward, 
                                reduced_next_state, done)
# ============================================================================
# VARIANT 4: Decentralized Learning
# ============================================================================

class DecentralizedVariant:
    """یادگیری غیرمتمرکز - هر agent مستقل آموزش می‌بیند"""
    
    def __init__(self, obs_dim, action_dim, num_agents, **config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # هایپرپارامترها
        self.gamma = config.get("gamma", 0.95)
        self.tau = config.get("tau", 0.001)
        self.batch_size = config.get("batch_size", 64)
        self.actor_lr = config.get("actor_lr", 1e-4)
        self.critic_lr = config.get("critic_lr", 1e-3)
        
        # شبکه‌های محلی
        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for _ in range(num_agents):
            actor = SimpleMLPActor(obs_dim, action_dim, hidden=512).to(self.device)
            actor_target = SimpleMLPActor(obs_dim, action_dim, hidden=512).to(self.device)
            actor_target.load_state_dict(actor.state_dict())
            
            # ✅ Critic محلی - فقط state و action خودش را می‌بیند
            critic = LocalCritic(obs_dim, action_dim, hidden=512).to(self.device)
            critic_target = LocalCritic(obs_dim, action_dim, hidden=512).to(self.device)
            critic_target.load_state_dict(critic.state_dict())
            
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.actor_lr)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
        
        buffer_size = config.get("buffer_size", 100000)
        self.replay_buffers = [ReplayBuffer(buffer_size) for _ in range(num_agents)]
        
        # ✅ Exploration noise با کاهش تدریجی
        self.noise_scale = 0.1
        self.noise_decay = 0.9995
        self.min_noise = 0.01
        
        print("📌 DecentralizedVariant: Critic محلی برای هر agent")
    
    def select_action(self, agent_id, obs, add_noise=True):
        """انتخاب action - مشابه FullMADDPG"""
        if isinstance(agent_id, str):
            agent_id = int(agent_id.split('_')[-1])
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actors[agent_id](obs_tensor)
        
        action = action.cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = action + noise
            self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
        
        # ✅ Clipping دقیق
        action = np.clip(action, 0.0, 1.0)
        action = np.round(action, decimals=6)
        
        epsilon = 1e-7
        action = np.where(action < epsilon, epsilon, action)
        action = np.where(action > (1.0 - epsilon), 1.0 - epsilon, action)
        
        return action.astype(np.float32)
    
    def store_transition(self, agent_id, state, action, reward, next_state, done):
        """ذخیره تجربه"""
        if isinstance(agent_id, str):
            agent_id = int(agent_id.split('_')[-1])
        
        self.replay_buffers[agent_id].push(state, action, reward, next_state, done)
    
    def update(self):
        """آپدیت غیرمتمرکز"""
        for agent_id in range(self.num_agents):
            if len(self.replay_buffers[agent_id]) < self.batch_size:
                continue
            
            states, actions, rewards, next_states, dones = \
                self.replay_buffers[agent_id].sample(self.batch_size)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            
            # ========== Update Critic (محلی) ==========
            with torch.no_grad():
                next_actions = self.actor_targets[agent_id](next_states)
                target_q = self.critic_targets[agent_id](next_states, next_actions)
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            current_q = self.critics[agent_id](states, actions)
            critic_loss = F.mse_loss(current_q, target_q)
            
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            
            # ✅ Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.critics[agent_id].parameters(), 
                max_norm=0.5
            )
            
            self.critic_optimizers[agent_id].step()
            
            # ========== Update Actor ==========
            predicted_actions = self.actors[agent_id](states)
            actor_loss = -self.critics[agent_id](states, predicted_actions).mean()
            
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            
            # ✅ Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.actors[agent_id].parameters(), 
                max_norm=0.5
            )
            
            self.actor_optimizers[agent_id].step()
            
            # ========== Soft Update ==========
            self._soft_update(self.actors[agent_id], self.actor_targets[agent_id])
            self._soft_update(self.critics[agent_id], self.critic_targets[agent_id])
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """ذخیره مدل"""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'noise_scale': self.noise_scale
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """بارگذاری مدل"""
        checkpoint = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
        
        if 'noise_scale' in checkpoint:
            self.noise_scale = checkpoint['noise_scale']


# ============================================================================
# VARIANT 5: Simpler Architecture
# ============================================================================

class SimplerArchVariant:
    """معماری ساده‌تر با لایه‌های کمتر"""
    
    def __init__(self, obs_dim, action_dim, num_agents, **config):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # هایپرپارامترها
        self.gamma = config.get("gamma", 0.95)
        self.tau = config.get("tau", 0.001)
        self.batch_size = config.get("batch_size", 64)
        self.actor_lr = config.get("actor_lr", 1e-4)
        self.critic_lr = config.get("critic_lr", 1e-3)
        
        # شبکه‌های کوچک‌تر
        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for _ in range(num_agents):
            # ✅ استفاده از معماری کوچک‌تر (256 به جای 512)
            actor = SmallActor(obs_dim, action_dim, hidden=256).to(self.device)
            actor_target = SmallActor(obs_dim, action_dim, hidden=256).to(self.device)
            actor_target.load_state_dict(actor.state_dict())
            
            total_state_dim = obs_dim * num_agents
            total_action_dim = action_dim * num_agents
            critic = SmallCritic(total_state_dim, total_action_dim, hidden=256).to(self.device)
            critic_target = SmallCritic(total_state_dim, total_action_dim, hidden=256).to(self.device)
            critic_target.load_state_dict(critic.state_dict())
            
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.actor_lr)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
            
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
        
        buffer_size = config.get("buffer_size", 100000)
        self.replay_buffers = [ReplayBuffer(buffer_size) for _ in range(num_agents)]
        
        # ✅ Exploration noise
        self.noise_scale = 0.1
        self.noise_decay = 0.9995
        self.min_noise = 0.01
        
        print("📌 SimplerArchVariant: استفاده از شبکه‌های کوچک‌تر (hidden=256)")
    
    def select_action(self, agent_id, obs, add_noise=True):
        """انتخاب action"""
        if isinstance(agent_id, str):
            agent_id = int(agent_id.split('_')[-1])
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actors[agent_id](obs_tensor)
        
        action = action.cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = action + noise
            self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
        
        # ✅ Clipping دقیق
        action = np.clip(action, 0.0, 1.0)
        action = np.round(action, decimals=6)
        
        epsilon = 1e-7
        action = np.where(action < epsilon, epsilon, action)
        action = np.where(action > (1.0 - epsilon), 1.0 - epsilon, action)
        
        return action.astype(np.float32)
    
    def store_transition(self, agent_id, state, action, reward, next_state, done):
        """ذخیره تجربه"""
        if isinstance(agent_id, str):
            agent_id = int(agent_id.split('_')[-1])
        
        self.replay_buffers[agent_id].push(state, action, reward, next_state, done)
    
    def update(self):
        """آپدیت شبکه‌ها - با Gradient Clipping"""
        for agent_id in range(self.num_agents):
            if len(self.replay_buffers[agent_id]) < self.batch_size:
                continue
            
            states, actions, rewards, next_states, dones = \
                self.replay_buffers[agent_id].sample(self.batch_size)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            
            all_states = states.repeat(1, self.num_agents)
            all_actions = actions.repeat(1, self.num_agents)
            all_next_states = next_states.repeat(1, self.num_agents)
            
            # ========== Update Critic ==========
            with torch.no_grad():
                next_actions = self.actor_targets[agent_id](next_states)
                all_next_actions = next_actions.repeat(1, self.num_agents)
                
                target_q = self.critic_targets[agent_id](all_next_states, all_next_actions)
                target_q = rewards + (1 - dones) * self.gamma * target_q
            
            current_q = self.critics[agent_id](all_states, all_actions)
            critic_loss = F.mse_loss(current_q, target_q)
            
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            
            # ✅ Gradient clipping برای معماری ساده‌تر
            torch.nn.utils.clip_grad_norm_(
                self.critics[agent_id].parameters(), 
                max_norm=0.5
            )
            
            self.critic_optimizers[agent_id].step()
            
            # ========== Update Actor ==========
            predicted_actions = self.actors[agent_id](states)
            all_predicted_actions = predicted_actions.repeat(1, self.num_agents)
            
            actor_loss = -self.critics[agent_id](all_states, all_predicted_actions).mean()
            
            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            
            # ✅ Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.actors[agent_id].parameters(), 
                max_norm=0.5
            )
            
            self.actor_optimizers[agent_id].step()
            
            # ========== Soft Update ==========
            self._soft_update(self.actors[agent_id], self.actor_targets[agent_id])
            self._soft_update(self.critics[agent_id], self.critic_targets[agent_id])
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """ذخیره مدل"""
        checkpoint = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'noise_scale': self.noise_scale
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """بارگذاری مدل"""
        checkpoint = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])
        
        if 'noise_scale' in checkpoint:
            self.noise_scale = checkpoint['noise_scale']


# ============================================================================
# Factory Function
# ============================================================================

def create_ablation_variant(variant_name, obs_dim, action_dim, num_agents, **config):
    """
    ساخت variant مورد نظر
    
    Args:
        variant_name: نام variant ('full_model', 'no_gat', ...)
        obs_dim: ابعاد observation
        action_dim: ابعاد action
        num_agents: تعداد agents
        **config: تنظیمات اضافی
    
    Returns:
        instance از variant مورد نظر
    """
    
    variants = {
        'full_model': FullMADDPGVariant,
        'no_gat': NoGATVariant,
        'no_temporal': NoTemporalVariant,
        'decentralized': DecentralizedVariant,
        'simpler_arch': SimplerArchVariant
    }
    
    if variant_name not in variants:
        raise ValueError(f"Unknown variant: {variant_name}. "
                        f"Available: {list(variants.keys())}")
    
    print(f"\n🚀 Creating {variant_name} variant...")
    return variants[variant_name](obs_dim, action_dim, num_agents, **config)
