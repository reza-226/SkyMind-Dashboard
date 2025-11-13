"""
تست برای بررسی خروجی محیط
"""
from core.env_multi import MultiUAVEnv
import numpy as np

env = MultiUAVEnv(n_agents=3, n_users=5, dt=0.1, area_size=1000.0)

# Reset
states = env.reset()

print("\n" + "="*70)
print("🔍 بررسی خروجی env.reset():")
print("="*70)
print(f"Type: {type(states)}")
print(f"Content:\n{states}")

if isinstance(states, dict):
    print("\n📦 Keys:")
    for key, value in states.items():
        print(f"  {key}: {type(value)} - shape: {np.array(value).shape if hasattr(value, '__len__') else 'N/A'}")
else:
    print(f"\n📦 Shape: {np.array(states).shape}")
    print(f"First element: {states[0]}")

# Step
actions = np.random.randn(3, 4)  # 3 agents, 4 actions each
next_states, rewards, dones, info = env.step(actions)

print("\n" + "="*70)
print("🔍 بررسی خروجی env.step():")
print("="*70)
print(f"next_states type: {type(next_states)}")
print(f"rewards type: {type(rewards)}")
print(f"dones type: {type(dones)}")
print(f"info type: {type(info)}")

if isinstance(next_states, dict):
    print("\n📦 next_states keys:")
    for key, value in next_states.items():
        print(f"  {key}: {type(value)}")
