# test_state_dim.py
import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from environments.uav_mec_env import UAVMECEnvironment

print("✅ Import successful!")

env = UAVMECEnvironment(
    num_uavs=8,
    num_devices=10,
    num_edge_servers=3,
    grid_size=1000.0,
    max_steps=100
)

print("✅ Environment created!")

state = env.reset()

print(f"\n📊 Environment State Information:")
print(f"✅ State type: {type(state)}")
print(f"✅ State shape: {state.shape}")
print(f"✅ State dimension: {len(state)}")
print(f"\n🎯 Use this value for state_dim: {len(state)}")

# ساخت یک action نمونه به صورت دستی
test_action = np.array([
    0,      # offload choice (0-4)
    0.5,    # cpu allocation [0,1]
    0.33,   # bandwidth[0]
    0.33,   # bandwidth[1]
    0.34,   # bandwidth[2]
    1.0,    # move_x
    1.0     # move_y
], dtype=np.float32)

print(f"\n📊 Action Information:")
print(f"✅ Action shape: {test_action.shape}")
print(f"✅ Action: {test_action}")

try:
    next_state, reward, done, info = env.step(test_action)
    print(f"\n✅ Step successful!")
    print(f"   Next state shape: {next_state.shape}")
    print(f"   Reward: {reward:.2f}")
    print(f"   Done: {done}")
except Exception as e:
    print(f"\n❌ Step failed: {e}")
    import traceback
    traceback.print_exc()
