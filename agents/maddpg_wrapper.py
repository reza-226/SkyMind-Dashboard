"""
agents/maddpg_wrapper.py
========================
Unified Wrapper برای هر دو نوع Agent
"""

import numpy as np
import torch
from collections import deque
import random


# ==================== Replay Buffer ====================
class MultiAgentReplayBuffer:
    """Replay Buffer مشترک برای Multi-Agent"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, states, actions, rewards, next_states, dones):
        """
        states: (n_agents, obs_dim) یا dict
        actions: (n_agents, act_dim)
        rewards: (n_agents,) یا scalar
        """
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size):
        """برگرداندن batch از transitions"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            states,    # list of states
            actions,   # list of actions
            rewards,   # list of rewards
            next_states,  # list of next_states
            dones      # list of dones
        )
    
    def __len__(self):
        return len(self.buffer)


# ==================== Unified MADDPG Agent ====================
class MADDPGAgent:
    """
    Wrapper یکپارچه برای agent_maddpg.py و agent_maddpg_multi.py
    """
    
    def __init__(self, n_agents, obs_dim, act_dim, 
                 hidden_dim=256, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.01, buffer_size=100000, device='cpu'):
        """
        Args:
            n_agents: تعداد agents
            obs_dim: بعد observation برای هر agent
            act_dim: بعد action برای هر agent
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        
        # تشخیص نوع Agent و import
        self.agent_type = self._detect_and_import_agent()
        
        # ایجاد agents
        if self.agent_type == 'multi':
            # استفاده از agent_maddpg_multi.py
            from agents.agent_maddpg_multi import MADDPG_Agent
            
            self.agents = [
                MADDPG_Agent(
                    state_dim=obs_dim,
                    action_dim=act_dim,
                    n_agents=n_agents,
                    lr=lr_actor,
                    gamma=gamma,
                    tau=tau,
                    device=device
                ) for _ in range(n_agents)
            ]
            
            # Replay buffer مشترک
            self.replay_buffer = MultiAgentReplayBuffer(buffer_size)
            
            print(f"✅ استفاده از agent_maddpg_multi.py ({n_agents} agents)")
            
        else:  # 'single'
            # استفاده از agent_maddpg.py
            from agents.agent_maddpg import MADDPGAgent as SingleAgent
            
            self.agents = [
                SingleAgent(
                    state_dim=obs_dim,
                    action_dim=act_dim,
                    lr=lr_actor,
                    gamma=gamma,
                    tau=tau
                ) for _ in range(n_agents)
            ]
            
            # این agent خود replay buffer دارد
            self.replay_buffer = None
            
            print(f"✅ استفاده از agent_maddpg.py ({n_agents} agents)")
    
    def _detect_and_import_agent(self):
        """تشخیص اینکه کدام agent موجود است"""
        try:
            from agents.agent_maddpg_multi import MADDPG_Agent
            return 'multi'
        except ImportError:
            pass
        
        try:
            from agents.agent_maddpg import MADDPGAgent
            return 'single'
        except ImportError:
            raise ImportError(
                "❌ هیچ کدام از agent files موجود نیست:\n"
                "  - agents/agent_maddpg_multi.py\n"
                "  - agents/agent_maddpg.py"
            )
    
    def select_action(self, observations, add_noise=True):
        """
        انتخاب action برای همه agents
        
        Args:
            observations: (n_agents, obs_dim) یا dict یا list
            add_noise: آیا noise اضافه شود (برای exploration)
        
        Returns:
            actions: (n_agents, act_dim) numpy array
        """
        # نرمال‌سازی input
        obs_array = self._normalize_observations(observations)
        
        actions = []
        
        if self.agent_type == 'multi':
            # agent_maddpg_multi: continuous actions
            noise_scale = 0.1 if add_noise else 0.0
            
            for i, agent in enumerate(self.agents):
                action = agent.act(obs_array[i], noise_scale=noise_scale)
                actions.append(action)
        
        else:  # 'single'
            # agent_maddpg: discrete actions
            for i, agent in enumerate(self.agents):
                action = agent.select_action(obs_array[i])
                
                # تبدیل discrete به continuous (فرض: 3 اکشن -> [-1, 0, 1])
                if action == 0:
                    continuous_action = np.array([-1.0, 0.0])
                elif action == 1:
                    continuous_action = np.array([0.0, 0.0])
                else:  # action == 2
                    continuous_action = np.array([1.0, 0.0])
                
                actions.append(continuous_action)
        
        return np.array(actions)
    
    def store_transition(self, states, actions, rewards, next_states, dones):
        """ذخیره transition در replay buffer"""
        if self.agent_type == 'multi':
            # ذخیره در buffer مشترک
            self.replay_buffer.push(states, actions, rewards, next_states, dones)
        else:
            # هر agent خودش ذخیره می‌کند
            states_array = self._normalize_observations(states)
            next_states_array = self._normalize_observations(next_states)
            
            # نرمال‌سازی rewards
            if isinstance(rewards, (int, float)):
                rewards_array = np.array([rewards] * self.n_agents)
            else:
                rewards_array = np.array(rewards)
            
            # نرمال‌سازی dones
            if isinstance(dones, bool):
                dones_array = np.array([dones] * self.n_agents)
            else:
                dones_array = np.array(dones)
            
            for i, agent in enumerate(self.agents):
                # برای agent_maddpg: action باید scalar باشد
                if len(actions.shape) == 2:
                    # تبدیل continuous به discrete
                    action_scalar = self._continuous_to_discrete(actions[i])
                else:
                    action_scalar = actions[i]
                
                agent.replay_buffer.push(
                    states_array[i],
                    action_scalar,
                    rewards_array[i],
                    next_states_array[i],
                    dones_array[i]
                )
    
    def train_step(self, batch_size=128):
        """یک گام آموزش"""
        if self.agent_type == 'multi':
            # آموزش با buffer مشترک
            losses = []
            for agent in self.agents:
                loss = agent.update(
                    self.replay_buffer,
                    other_agents=self.agents,
                    batch_size=batch_size
                )
                if loss is not None:
                    losses.append(loss)
            
            return losses if losses else None
        
        else:  # 'single'
            # هر agent جداگانه train می‌شود
            for agent in self.agents:
                agent.train(batch_size=batch_size)
            
            return None
    
    def save_model(self, filepath):
        """ذخیره مدل‌ها"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.agent_type == 'multi':
            state_dict = {
                f'agent_{i}': {
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'target_actor': agent.target_actor.state_dict(),
                    'target_critic': agent.target_critic.state_dict()
                }
                for i, agent in enumerate(self.agents)
            }
        else:
            state_dict = {
                f'agent_{i}': {
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'target_actor': agent.target_actor.state_dict(),
                    'target_critic': agent.target_critic.state_dict()
                }
                for i, agent in enumerate(self.agents)
            }
        
        torch.save(state_dict, filepath)
    
    def load_model(self, filepath):
        """بارگذاری مدل‌ها"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        for i, agent in enumerate(self.agents):
            agent_key = f'agent_{i}'
            if agent_key in checkpoint:
                agent.actor.load_state_dict(checkpoint[agent_key]['actor'])
                agent.critic.load_state_dict(checkpoint[agent_key]['critic'])
                agent.target_actor.load_state_dict(checkpoint[agent_key]['target_actor'])
                agent.target_critic.load_state_dict(checkpoint[agent_key]['target_critic'])
    
    # ========== Helper Methods ==========
    def _normalize_observations(self, obs):
        """تبدیل observations به numpy array"""
        if isinstance(obs, dict):
            # فرض: obs = {0: arr, 1: arr, ...}
            return np.array([obs[i] for i in range(self.n_agents)])
        elif isinstance(obs, list):
            return np.array(obs)
        elif isinstance(obs, np.ndarray):
            if len(obs.shape) == 1:
                # یک observation -> تکرار برای همه agents
                return np.tile(obs, (self.n_agents, 1))
            return obs
        else:
            raise ValueError(f"فرمت observation نامعتبر: {type(obs)}")
    
    def _continuous_to_discrete(self, action):
        """تبدیل continuous action به discrete"""
        # فرض: action = [vx, vy]
        if action[0] < -0.3:
            return 0  # left
        elif action[0] > 0.3:
            return 2  # right
        else:
            return 1  # stay
