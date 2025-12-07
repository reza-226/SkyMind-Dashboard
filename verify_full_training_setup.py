# verify_full_training_setup.py
import sys
sys.path.append('environments')
sys.path.append('agents')

from uav_mec_env import UAVMECEnvironment
from maddpg_agent import MADDPGAgent
import torch
import numpy as np

print("="*70)
print("🔍 Complete Training Setup Verification")
print("="*70)

# ============================================
# 1. محیط
# ============================================
print("\n📋 Step 1: Environment Setup")
env = UAVMECEnvironment(
    num_uavs=5,
    num_devices=10,
    num_edge_servers=2,
    grid_size=1000.0,
    max_steps=100
)

state = env.reset()
state_dim = len(state)
action_dim = 7

print(f"   ✅ Environment initialized")
print(f"   State dimension: {state_dim}")
print(f"   Action dimension: {action_dim}")
print(f"   Number of UAVs: {env.num_uavs}")

# ============================================
# 2. Agent Setup
# ============================================
print("\n📋 Step 2: Agent Setup")

total_state_dim = state_dim * env.num_uavs
total_action_dim = action_dim * env.num_uavs

print(f"   Total state dim: {total_state_dim}")
print(f"   Total action dim: {total_action_dim}")

try:
    agent = MADDPGAgent(
        agent_id=0,
        state_dim=state_dim,
        action_dim=action_dim,
        total_state_dim=total_state_dim,
        total_action_dim=total_action_dim,
        num_agents=env.num_uavs
    )
    print(f"   ✅ Agent created successfully")
except Exception as e:
    print(f"   ❌ Agent creation failed: {e}")
    exit(1)

# ============================================
# 3. Actor Network بررسی
# ============================================
print("\n📋 Step 3: Actor Network Verification")

try:
    # بررسی ساختار
    print(f"   Actor type: {type(agent.actor)}")
    
    # بررسی input layer
    if hasattr(agent.actor, 'fc1'):
        fc1_in = agent.actor.fc1.in_features
        fc1_out = agent.actor.fc1.out_features
        print(f"   fc1: {fc1_in} → {fc1_out}")
        
        if fc1_in != state_dim:
            print(f"   ⚠️  WARNING: fc1 input ({fc1_in}) != state_dim ({state_dim})")
        else:
            print(f"   ✅ fc1 input matches state_dim")
    
    # تست forward
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    print(f"   Test input shape: {state_tensor.shape}")
    
    with torch.no_grad():
        action = agent.actor(state_tensor)
    
    print(f"   ✅ Forward pass successful")
    print(f"   Output shape: {action.shape}")
    print(f"   Expected: torch.Size([1, {action_dim}])")
    
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================
# 4. محیط API بررسی
# ============================================
print("\n📋 Step 4: Environment API Check")

# بررسی methods
print(f"   Available methods:")
env_methods = [m for m in dir(env) if not m.startswith('_') and callable(getattr(env, m))]
print(f"   {env_methods[:10]}")

# بررسی step signature
import inspect
if hasattr(env, 'step'):
    sig = inspect.signature(env.step)
    print(f"\n   env.step() signature: {sig}")
else:
    print(f"\n   ⚠️  env.step() not found!")

# تست یک step
print(f"\n   Testing env.step()...")
try:
    # فرض: action باید array یا list باشد
    test_action = np.random.randn(action_dim)
    print(f"   Test action shape: {test_action.shape}")
    
    # تلاش برای step
    result = env.step(test_action)
    print(f"   ✅ env.step() executed")
    print(f"   Result type: {type(result)}")
    
    if isinstance(result, tuple):
        print(f"   Result length: {len(result)}")
        if len(result) >= 2:
            next_state, reward = result[0], result[1]
            print(f"   next_state type: {type(next_state)}")
            print(f"   next_state shape: {next_state.shape if hasattr(next_state, 'shape') else len(next_state)}")
            print(f"   reward: {reward}")
            
except Exception as e:
    print(f"   ❌ env.step() failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("🎯 Verification Summary:")
print(f"   State dimension: {state_dim}")
print(f"   Action dimension: {action_dim}")
print(f"   Agent created: ✅")
print(f"   Forward pass: ✅")
print("="*70)
