# ============================================================
# File: agents/agent_maddpg_multi.py
# SkyMind‑TPSG : Multi‑Agent DDPG Implementation (Final Fixed)
# Compatible with train_multi.py (batch_size arg)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

# ------------------------------------------------------------
# Actor Network: decision policy for (v, θ, o, f)
# ------------------------------------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))


# ------------------------------------------------------------
# Critic Network: evaluates Q(s,a)
# ------------------------------------------------------------
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + (action_dim * n_agents), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ------------------------------------------------------------
# Replay Buffer
# ------------------------------------------------------------
Experience = namedtuple(
    'Experience', field_names=['states', 'actions', 'rewards', 'next_states', 'dones']
)

class ReplayBuffer:
    def __init__(self, buffer_size=int(1e6), batch_size=128):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.FloatTensor(np.vstack([e.states for e in experiences]))
        actions = torch.FloatTensor(np.vstack([e.actions for e in experiences]))
        rewards = torch.FloatTensor(np.vstack([e.rewards for e in experiences]))
        next_states = torch.FloatTensor(np.vstack([e.next_states for e in experiences]))
        dones = torch.FloatTensor(np.vstack([e.dones for e in experiences]))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# ------------------------------------------------------------
# Ornstein–Uhlenbeck noise for continuous exploration
# ------------------------------------------------------------
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# ------------------------------------------------------------
# Main MADDPG Agent Class
# ------------------------------------------------------------
class MADDPG_Agent:
    def __init__(self, state_dim, action_dim, n_agents, lr=1e-3, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = 0.01

        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim, action_dim, n_agents)
        self.target_actor = ActorNetwork(state_dim, action_dim)
        self.target_critic = CriticNetwork(state_dim, action_dim, n_agents)

        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.noise = OUNoise(action_dim)

    # --------------------------------------------------------
    def act(self, state, noise_scale=0.1):
        self.actor.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state_t).squeeze(0).numpy()
        self.actor.train()
        action += self.noise.sample() * noise_scale
        return np.clip(action, -1, 1)

    # --------------------------------------------------------
    def update(self, replay_buffer, other_agents, batch_size=None):
        """
        Performs MADDPG update using replay buffer.
        'batch_size' is optional for backward compatibility with train_multi.py
        """
        if replay_buffer is None or len(replay_buffer) < (replay_buffer.batch_size if hasattr(replay_buffer, 'batch_size') else (batch_size or 128)):
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample()

        # ---- Critic target ----
        with torch.no_grad():
            next_actions = []
            for ag in other_agents:
                next_actions.append(ag.target_actor(next_states))
            next_actions = torch.cat(next_actions, dim=1)

            q_target_next = self.target_critic(next_states, next_actions)
            q_targets = rewards + self.gamma * q_target_next * (1 - dones)

        # ---- Critic loss ----
        pred_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(pred_q, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Actor loss ----
        pred_actions = self.actor(states)
        all_pred_actions = torch.cat([pred_actions for _ in range(self.n_agents)], dim=1)
        actor_loss = -self.critic(states, all_pred_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Soft update ----
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

    # --------------------------------------------------------
    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
