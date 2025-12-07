# check_state_dim.py
import numpy as np
from environments.uav_mec_env import UAVMECEnvironment

print("="*60)
print("🔍 Checking ACTUAL State Dimensions")
print("="*60)

env = UAVMECEnvironment()

# Reset و بررسی
state = env.reset()
if isinstance(state, tuple):
    state = state[0]

print(f"\n📊 State Information:")
print(f"   Type: {type(state)}")
print(f"   Shape: {state.shape if hasattr(state, 'shape') else len(state)}")
print(f"   Actual Dimension: {len(state) if isinstance(state, np.ndarray) else 'N/A'}")

# چند episode تست کنیم
print(f"\n🧪 Testing multiple resets:")
for i in range(5):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    dim = len(state) if isinstance(state, np.ndarray) else state.shape[0]
    print(f"   Reset {i+1}: dimension = {dim}")

# یک step هم تست کنیم
print(f"\n🚶 Testing one step:")
state = env.reset()
if isinstance(state, tuple):
    state = state[0]
    
action = np.random.uniform(-1, 1, size=7)
next_state, reward, done, info = env.step(action)

if isinstance(next_state, tuple):
    next_state = next_state[0]

print(f"   Next state dimension: {len(next_state)}")

print(f"\n{'='*60}")
print(f"✅ CONFIRMED STATE DIMENSION: {len(state)}")
print(f"{'='*60}\n")
