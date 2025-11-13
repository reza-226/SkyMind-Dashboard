import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    """شبکه Actor برای انتخاب اکشن"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """شبکه Critic برای ارزیابی حالت-اکشن (Centralized)"""
    def __init__(self, total_state_dim, total_action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states, actions):
        """
        states: (batch, total_state_dim)
        actions: (batch, total_action_dim)
        """
        x = torch.cat([states, actions], dim=1)
        return self.net(x)


class MADDPG_Agent:
    """یک Agent MADDPG با Centralized Critic"""
    def __init__(self, state_dim, action_dim, n_agents, 
                 lr=1e-4, gamma=0.99, tau=0.01, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # ابعاد کل
        self.total_state_dim = state_dim * n_agents
        self.total_action_dim = action_dim * n_agents
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(self.total_state_dim, self.total_action_dim).to(device)
        self.target_critic = Critic(self.total_state_dim, self.total_action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr*10)
    
    def act(self, state, noise_scale=0.0):
        """انتخاب action"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        
        if noise_scale > 0:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def update(self, replay_buffer, other_agents=None, batch_size=128):
        """بروزرسانی Actor و Critic"""
        if len(replay_buffer) < batch_size:
            return None
        
        # Sample batch
        batch = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # تبدیل به numpy
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Reshape states اگر 3D است
        if len(states.shape) == 3:
            batch_size_actual = states.shape[0]
            states = states.reshape(batch_size_actual, -1)
        
        if len(next_states.shape) == 3:
            batch_size_actual = next_states.shape[0]
            next_states = next_states.reshape(batch_size_actual, -1)
        
        # Flatten actions
        if len(actions.shape) == 3:
            batch_size_actual = actions.shape[0]
            actions = actions.reshape(batch_size_actual, -1)
        
        # Reshape rewards و dones
        if len(rewards.shape) == 2:
            rewards = rewards[:, 0]
        
        if len(dones.shape) == 2:
            dones = dones[:, 0]
        
        # تبدیل به tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # ========== Update Critic ==========
        with torch.no_grad():
            # محاسبه next_actions
            next_actions_list = []
            for i in range(self.n_agents):
                start_idx = i * self.state_dim
                end_idx = (i + 1) * self.state_dim
                agent_next_state = next_states[:, start_idx:end_idx]
                agent_next_action = self.target_actor(agent_next_state)
                next_actions_list.append(agent_next_action)
            
            next_actions = torch.cat(next_actions_list, dim=1)
            
            # Q_target
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Q_current
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # بروزرسانی Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ========== Update Actor ==========
        current_actions_list = []
        for i in range(self.n_agents):
            start_idx = i * self.state_dim
            end_idx = (i + 1) * self.state_dim
            agent_state = states[:, start_idx:end_idx]
            agent_action = self.actor(agent_state)
            current_actions_list.append(agent_action)
        
        current_actions = torch.cat(current_actions_list, dim=1)
        
        # Actor loss
        actor_loss = -self.critic(states, current_actions).mean()
        
        # بروزرسانی Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft Update
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict()
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
