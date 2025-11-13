"""
inspect_env_step.py
بررسی خروجی step و info
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.env_multi import MultiUAVEnv

print("=" * 70)
print("🔍 Inspecting MultiUAVEnv.step() Output")
print("=" * 70)

# ساخت محیط
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
state = env.reset()

print(f"\n✅ Environment initialized")
print(f"State keys: {state.keys()}")

# یک action تصادفی
actions = np.random.randn(3, 4)  # (n_agents, action_dim)
print(f"\n🎮 Taking random action with shape: {actions.shape}")

# اجرای step
step_output = env.step(actions)

print(f"\n📦 Step output length: {len(step_output)}")

if len(step_output) == 5:
    next_state, reward, done, truncated, info = step_output
    print("\n✅ Output format: (state, reward, done, truncated, info)")
elif len(step_output) == 4:
    next_state, reward, done, info = step_output
    print("\n✅ Output format: (state, reward, done, info)")
else:
    print(f"\n⚠️  Unexpected output length: {len(step_output)}")
    sys.exit(1)

# بررسی reward
print(f"\n💰 Reward:")
print(f"   Type: {type(reward)}")
if isinstance(reward, np.ndarray):
    print(f"   Shape: {reward.shape}")
    print(f"   Values: {reward}")
else:
    print(f"   Value: {reward}")

# بررسی done
print(f"\n🏁 Done: {done} (type: {type(done)})")

# بررسی info
print(f"\n📋 Info:")
print(f"   Type: {type(info)}")
if isinstance(info, dict):
    print(f"   Keys: {info.keys()}")
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: shape={value.shape}, sum={np.sum(value):.2e}")
        else:
            print(f"   {key}: {value}")
else:
    print(f"   Value: {info}")

print("\n" + "=" * 70)
