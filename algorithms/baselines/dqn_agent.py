# algorithms/baselines/dqn_agent.py

"""
DQN Agent for UAV Offloading
==============================
Deep Q-Network با Discrete Action Space
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Tuple


class DQNNetwork(nn.Module):
    """Q-Network برای DQN"""
    
    def __init__(self, state_dim=537, action_dim=5, hidden_dims=[512, 512]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


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


class DQNAgent:
    """
    DQN Agent برای UAV Offloading
    
    Action Space: 5 discrete actions
      0: Local processing
      1: Edge offloading
      2: Cloud offloading
      3: Fog offloading
      4: Ground offloading
    """
    
    def __init__(
        self,
        state_dim=537,
        action_dim=5,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        tau=0.005,
        buffer_size=100000,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Stats
        self.update_count = 0
        self.loss_history = []
    
    def select_action(self, state: np.ndarray, evaluation=False) -> Dict:
        """
        انتخاب action با ε-greedy
        
        Returns:
            action dict compatible با environment
        """
        # Ensure state is 1D
        if state.ndim > 1:
            state = state.flatten()
        
        if evaluation or random.random() > self.epsilon:
            # Exploitation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                offload_action = q_values.argmax(dim=1).item()
        else:
            # Exploration
            offload_action = random.randrange(self.action_dim)
        
        # ساخت action dict کامل
        action = {
            "offload": offload_action,
            "cpu": 0.7,  # مقدار ثابت
            "bandwidth": np.array([0.4, 0.3, 0.3]),
            "move": np.array([0.0, 0.0])
        }
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """ذخیره تجربه"""
        # فقط offload decision را ذخیره می‌کنیم
        if isinstance(action, dict):
            action = action["offload"]
        
        self.memory.push(
            state.flatten() if state.ndim > 1 else state,
            action,
            reward,
            next_state.flatten() if next_state.ndim > 1 else next_state,
            done
        )
    
    def train_step(self):
        """یک گام آموزش"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.update_count += 1
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def soft_update(self):
        """Soft update target network"""
        for target_param, param in zip(
            self.target_network.parameters(), 
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """ذخیره مدل"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'loss_history': self.loss_history
        }, filepath)
        print(f"✅ DQN model saved to {filepath}")
    
    def load(self, filepath):
        """بارگذاری مدل"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.update_count = checkpoint.get('update_count', 0)
        self.loss_history = checkpoint.get('loss_history', [])
        print(f"✅ DQN model loaded from {filepath}")


# ========================================
# Test Function
# ========================================

def test_dqn():
    """تست DQN Agent"""
    print("=" * 60)
    print("🧪 Testing DQN Agent")
    print("=" * 60)
    
    agent = DQNAgent(state_dim=537, action_dim=5)
    
    # Test action selection
    dummy_state = np.random.rand(537)
    action = agent.select_action(dummy_state)
    print(f"\n📊 Action selected:")
    print(f"   Offload: {action['offload']}")
    print(f"   CPU: {action['cpu']}")
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
    
    loss = agent.train_step()
    if loss is not None:
        print(f"   Loss: {loss:.4f}")
        print(f"   ✅ Training works!")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_dqn()
