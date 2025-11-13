"""
run_maddpg_experiment_FIXED.py
اجرای کامل MADDPG با پارامترهای صحیح - Version 2
"""

import numpy as np
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent))

from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent

print("=" * 70)
print("🚀 MADDPG Multi-Agent Experiment (Random Policy)")
print("=" * 70)

# ============================================================================
# Multi-Agent Wrapper
# ============================================================================
class MultiAgentMADDPG:
    """Wrapper برای مدیریت چند agent مستقل"""
    
    def __init__(self, state_dim, action_dim, n_agents):
        self.n_agents = n_agents
        self.action_dim = action_dim
        
        # ساخت یک agent برای هر UAV
        self.agents = [
            MADDPG_Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                lr=1e-4,
                gamma=0.95
            )
            for _ in range(n_agents)
        ]
        
        print(f"✅ Created {n_agents} independent agents")
    
    def act(self, state, noise_scale=0.0):
        """
        هر agent به state کامل دسترسی داره (centralized training)
        ولی action مستقل انتخاب می‌کنه
        """
        actions = []
        for agent in self.agents:
            action = agent.act(state, noise_scale)
            actions.append(action)
        
        # تبدیل به (n_agents, action_dim)
        return np.array(actions)

# ============================================================================
# محیط
# ============================================================================
env_config = {
    'n_agents': 3,
    'n_users': 10,
    'dt': 1.0,
    'area_size': 1000.0,
    'c1': 9.26e-4,
    'c2': 2250.0,
    'bandwidth': 1e6,
    'noise_power': 1e-10,
    'alpha_delay': 1.0,
    'beta_energy': 1e-6,
    'gamma_eff': 1e3
}

env = MultiUAVEnv(**env_config)
print(f"\n📋 Environment created successfully!")
print(f"   n_agents: {env_config['n_agents']}")
print(f"   n_users : {env_config['n_users']}")
print(f"   area_size: {env_config['area_size']}m")

# ============================================================================
# بررسی ساختار State
# ============================================================================
print("\n" + "=" * 70)
print("🔍 Inspecting State Structure")
print("=" * 70)

state = env.reset()
print(f"\nState type: {type(state)}")

if isinstance(state, dict):
    print(f"State keys: {state.keys()}")
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   {key}: type={type(value)}, value={value}")
    
    # تبدیل dict به vector مسطح
    state_vector = []
    for key in sorted(state.keys()):
        val = state[key]
        if isinstance(val, np.ndarray):
            state_vector.append(val.flatten())
        else:
            state_vector.append(np.array([val]).flatten())
    
    state_flat = np.concatenate(state_vector)
    state_dim = len(state_flat)
    
    print(f"\n✅ Flattened state dimension: {state_dim}")
    
elif isinstance(state, np.ndarray):
    state_dim = state.shape[0] if len(state.shape) == 1 else np.prod(state.shape)
    state_flat = state.flatten()
    print(f"\n✅ Array state dimension: {state_dim}")
    
else:
    print(f"⚠️  Unknown state type: {type(state)}")
    sys.exit(1)

# ============================================================================
# Agent
# ============================================================================
multi_agent = MultiAgentMADDPG(
    state_dim=state_dim,
    action_dim=4,
    n_agents=3
)

print(f"\n📋 Agent Configuration:")
print(f"   State dim: {state_dim}")
print(f"   Action dim: 4")
print(f"   N agents: 3")
print(f"\n⚠️  Using RANDOM policy (no pre-trained models)")

# ============================================================================
# Helper function برای تبدیل state
# ============================================================================
def state_to_vector(state):
    """تبدیل state (dict یا array) به vector"""
    if isinstance(state, dict):
        state_vector = []
        for key in sorted(state.keys()):
            val = state[key]
            if isinstance(val, np.ndarray):
                state_vector.append(val.flatten())
            else:
                state_vector.append(np.array([val]).flatten())
        return np.concatenate(state_vector)
    elif isinstance(state, np.ndarray):
        return state.flatten()
    else:
        return state

# Helper برای تبدیل به scalar
def to_scalar(value):
    """تبدیل هر نوع value به یک عدد scalar"""
    if isinstance(value, np.ndarray):
        return float(np.sum(value))  # یا np.mean(value)
    elif isinstance(value, (list, tuple)):
        return float(np.sum(value))
    else:
        return float(value)

# ============================================================================
# اجرای Episode
# ============================================================================
print("\n" + "=" * 70)
print("🎮 Running Episodes")
print("=" * 70)

n_episodes = 10
results = {
    'rewards': [],
    'delays': [],
    'energies': []
}

for ep in range(n_episodes):
    state = env.reset()
    state_vec = state_to_vector(state)
    
    episode_reward = 0.0
    episode_delay = 0.0
    episode_energy = 0.0
    done = False
    step = 0
    
    print(f"\n📍 Episode {ep + 1}/{n_episodes}")
    
    while not done and step < 100:
        # گرفتن action از multi-agent
        actions = multi_agent.act(state_vec, noise_scale=0.0)
        
        # اجرا در محیط
        step_result = env.step(actions)
        
        # بررسی تعداد خروجی‌ها
        if len(step_result) == 5:
            next_state, reward, done, truncated, info = step_result
        elif len(step_result) == 4:
            next_state, reward, done, info = step_result
            truncated = False
        else:
            print(f"⚠️  Unexpected step output length: {len(step_result)}")
            break
        
        # محاسبه متریک‌ها (تبدیل به scalar)
        reward_scalar = to_scalar(reward)
        episode_reward += reward_scalar
        
        # استخراج delay و energy از info
        if isinstance(info, dict):
            if 'delay' in info:
                episode_delay += to_scalar(info['delay'])
            if 'energy' in info:
                episode_energy += to_scalar(info['energy'])
        
        state = next_state
        state_vec = state_to_vector(state)
        step += 1
        
        if done or truncated:
            break
    
    results['rewards'].append(episode_reward)
    results['delays'].append(episode_delay)
    results['energies'].append(episode_energy)
    
    print(f"   Steps: {step}")
    print(f"   Total Reward: {episode_reward:.2f}")
    print(f"   Total Delay: {episode_delay:.2f}s")
    print(f"   Total Energy: {episode_energy:.2e}J")

# ============================================================================
# خلاصه نتایج
# ============================================================================
print("\n" + "=" * 70)
print("📊 Results Summary")
print("=" * 70)

for metric_name, values in results.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"\n{metric_name.upper()}:")
    print(f"   Mean: {mean_val:.2e}")
    print(f"   Std:  {std_val:.2e}")
    print(f"   Min:  {np.min(values):.2e}")
    print(f"   Max:  {np.max(values):.2e}")

print("\n✅ Experiment completed!")
print("\n💡 Note: Results are based on RANDOM policy.")
print("   Train the model with correct state_dim=38 for better results.")
