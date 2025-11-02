# ==========================================
# agents/agent_maddpg.py
# Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
# Adapted for SkyMind Environment (from Payannameh + ECORI paper)
# ==========================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from core.env import SkyMindEnv

# -----------------------------------------------------------
# Neural Network Architectures
# -----------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)

# -----------------------------------------------------------
# MADDPG Agent (Single agent simplified)
# -----------------------------------------------------------
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.95, tau=0.01):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)

        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.update_targets(tau=1.0)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state_tensor).detach().numpy()[0]
        return np.random.choice(len(probs), p=probs)

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        s, a, r, s2, d = self.replay_buffer.sample(batch_size)
        # Convert action to one-hot
        a_onehot = torch.zeros(batch_size, 3)
        a_onehot[range(batch_size), a] = 1

        # Critic Loss
        with torch.no_grad():
            target_action_probs = self.target_actor(s2)
            next_action = torch.multinomial(target_action_probs, 1)
            next_a_onehot = torch.zeros(batch_size, 3)
            next_a_onehot[range(batch_size), next_action.squeeze()] = 1
            target_q = self.target_critic(s2, next_a_onehot)
            y = r.unsqueeze(1) + self.gamma * (1 - d.unsqueeze(1)) * target_q

        q_val = self.critic(s, a_onehot)
        critic_loss = nn.MSELoss()(q_val, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Loss
        current_probs = self.actor(s)
        curr_actions = torch.multinomial(current_probs, 1)
        curr_onehot = torch.zeros(batch_size, 3)
        curr_onehot[range(batch_size), curr_actions.squeeze()] = 1
        actor_loss = -self.critic(s, curr_onehot).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_targets()

    def update_targets(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# -----------------------------------------------------------
# Training Loop (Demo)
# -----------------------------------------------------------
if __name__ == "__main__":
    env = SkyMindEnv()
    agent = MADDPGAgent(env.observation_space_dim(), len(env.actions))
    episodes = 5  # for quick demo

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for step in range(20):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
        print(f"Episode {ep+1} | Total reward: {total_reward:.3f}")
