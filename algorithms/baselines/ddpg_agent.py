# algorithms/baselines/ddpg_agent.py

"""
DDPG Agent for UAV Offloading
===============================
Deep Deterministic Policy Gradient با Continuous Action Space
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple


class ActorNetwork(nn.Module):
    """Actor Network - تولید continuous actions"""
    
    def __init__(self, state_dim=537, action_dim=6, hidden_dims=[512, 512]):
        super(ActorNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state):
        x = self.network(state)
        # tanh برای محدود کردن به [-1, 1]
        return torch.tanh(self.output(x))


class CriticNetwork(nn.Module):
    """Critic Network - ارزیابی (state, action) pair"""
    
    def __init__(self, state_dim=537, action_dim=6, hidden_dims=[512, 512]):
        super(CriticNetwork, self).__init__()
        
        # State processing
        self.state_layer = nn.Linear(state_dim, hidden_dims[0])
        
        # Combined processing
        layers = []
        prev_dim = hidden_dims[0] + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.combined_network = nn.Sequential(*layers)
        self.q_value = nn.Linear(prev_dim, 1)
    
    def forward(self, state, action):
        s = F.elu(self.state_layer(state))
        x = torch.cat([s, action], dim=1)
        x = self.combined_network(x)
        return self.q_value(x)


class OUNoise:
    """Ornstein-Uhlenbeck Noise برای exploration"""
    
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity=100000):
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


class DDPGAgent:
    """
    DDPG Agent برای UAV Offloading
    
    Continuous Action Space (6 dims):
      - CPU allocation: [0, 1]
      - Bandwidth allocation: [0, 1] x 3 (normalized)
      - Movement: [-max_step, max_step] x 2
    
    Note: Offload decision (discrete) با heuristic مشخص می‌شود
    """
    
    def __init__(
        self,
        state_dim=537,
        action_dim=6,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=64,
        max_movement=5.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_movement = max_movement
        self.device = torch.device(device)
        
        # Actor Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic Networks
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Noise
        self.noise = OUNoise(action_dim)
        
        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Stats
        self.update_count = 0
        self.actor_loss_history = []
        self.critic_loss_history = []
    
    def select_action(self, state: np.ndarray, evaluation=False) -> Dict:
        """
        انتخاب action
        
        Returns:
            action dict compatible با environment
        """
        # Ensure state is 1D
        if state.ndim > 1:
            state = state.flatten()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # Add noise for exploration
        if not evaluation:
            noise = self.noise.sample()
            action = np.clip(action + noise * 0.1, -1, 1)
        
        # Decode action
        return self._decode_action(action, state)
    
    def _decode_action(self, raw_action: np.ndarray, state: np.ndarray) -> Dict:
        """
        تبدیل raw continuous action به action dict
        
        raw_action shape: (6,) in [-1, 1]
        """
        # CPU allocation: [0, 1]
        cpu = (raw_action[0] + 1) / 2
        
        # Bandwidth allocation: normalize to sum=1
        bandwidth_raw = (raw_action[1:4] + 1) / 2
        bandwidth = bandwidth_raw / (bandwidth_raw.sum() + 1e-8)
        
        # Movement: [-max_step, max_step]
        move = raw_action[4:6] * self.max_movement
        
        # Offload decision با heuristic (براساس CPU capacity)
        cpu_capacity = state[5] if len(state) > 5 else 0.5
        if cpu_capacity > 0.7:
            offload = 0  # Local
        elif cpu_capacity > 0.4:
            offload = 1  # Edge
        else:
            offload = 2  # Cloud
        
        return {
            "offload": offload,
            "cpu": float(cpu),
            "bandwidth": bandwidth,
            "move": move
        }
    
    def store_transition(self, state, action, reward, next_state, done):
        """ذخیره تجربه"""
        # فقط continuous action را ذخیره می‌کنیم
        if isinstance(action, dict):
            # Reconstruct raw action
            cpu_raw = action["cpu"] * 2 - 1
            bw_raw = action["bandwidth"] * 2 - 1
            move_raw = action["move"] / self.max_movement
            raw_action = np.concatenate([[cpu_raw], bw_raw, move_raw])
        else:
            raw_action = action
        
        self.memory.push(
            state.flatten() if state.ndim > 1 else state,
            raw_action,
            reward,
            next_state.flatten() if next_state.ndim > 1 else next_state,
            done
        )
    
    def train_step(self):
        """یک گام آموزش"""
        if len(self.memory) < self.batch_size:
            return None, None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # ========== Update Critic ==========
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions).squeeze(1)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # ========== Update Actor ==========
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # ========== Soft Update Target Networks ==========
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        self.update_count += 1
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()
    
    def soft_update(self, source, target):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """ذخیره مدل"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'update_count': self.update_count,
            'actor_loss_history': self.actor_loss_history,
            'critic_loss_history': self.critic_loss_history
        }, filepath)
        print(f"✅ DDPG model saved to {filepath}")
    
    def load(self, filepath):
        """بارگذاری مدل"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.update_count = checkpoint.get('update_count', 0)
        self.actor_loss_history = checkpoint.get('actor_loss_history', [])
        self.critic_loss_history = checkpoint.get('critic_loss_history', [])
        print(f"✅ DDPG model loaded from {filepath}")


# ========================================
# Test Function
# ========================================

def test_ddpg():
    """تست DDPG Agent"""
    print("=" * 60)
    print("🧪 Testing DDPG Agent")
    print("=" * 60)
    
    agent = DDPGAgent(state_dim=537, action_dim=6)
    
    # Test action selection
    dummy_state = np.random.rand(537)
    action = agent.select_action(dummy_state)
    print(f"\n📊 Action selected:")
    print(f"   Offload: {action['offload']}")
    print(f"   CPU: {action['cpu']:.3f}")
    print(f"   Bandwidth: {action['bandwidth']}")
    print(f"   Move: {action['move']}")
    print(f"   ✅ Action selection works!")
    
    # Test training
    print(f"\n🔄 Testing training...")
    for _ in range(100):
        state = np.random.rand(537)
        action = agent.select_action(state)
        reward = np.random.rand()
        next_state = np.random.rand(537)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
    
    actor_loss, critic_loss = agent.train_step()
    if actor_loss is not None:
        print(f"   Actor Loss: {actor_loss:.4f}")
        print(f"   Critic Loss: {critic_loss:.4f}")
        print(f"   ✅ Training works!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_ddpg()
