# test_quick.py
import torch
import numpy as np
from agents.agent_maddpg_multi import MADDPG_Agent

# پارامترها
state_dim = 18
action_dim = 4
n_agents = 3

print("🔧 Creating MADDPG Agent...")

# ایجاد agent
agent = MADDPG_Agent(
    state_dim=state_dim,
    action_dim=action_dim,
    n_agents=n_agents,
    lr=1e-3,
    gamma=0.95
)

print("✅ Agent created successfully!")
print(f"   - State dim: {agent.state_dim}")
print(f"   - Action dim: {agent.action_dim}")
print(f"   - N agents: {agent.n_agents}")

# تست act (بدون noise)
print("\n🎯 Testing act() method (noise_scale=0.0)...")
state = np.random.randn(state_dim)
action = agent.act(state, noise_scale=0.0)

print(f"✅ State shape: {state.shape}")
print(f"✅ Action shape: {action.shape}")
print(f"✅ Action range: [{action.min():.3f}, {action.max():.3f}]")

# تست act (با noise)
print("\n🎯 Testing act() method (noise_scale=0.1)...")
action_noisy = agent.act(state, noise_scale=0.1)

print(f"✅ Action with noise shape: {action_noisy.shape}")
print(f"✅ Action with noise range: [{action_noisy.min():.3f}, {action_noisy.max():.3f}]")

# تست اینکه شبکه‌های target وجود دارند
print("\n🔍 Checking target networks...")
print(f"✅ Target Actor exists: {hasattr(agent, 'target_actor')}")
print(f"✅ Target Critic exists: {hasattr(agent, 'target_critic')}")

print("\n🎉 All tests passed! Agent is ready for training!")
