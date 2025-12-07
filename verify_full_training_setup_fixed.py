# verify_full_training_setup_fixed.py
import sys
import os

# اطمینان از import صحیح
sys.path.insert(0, os.path.join(os.getcwd(), 'environments'))
sys.path.insert(0, os.path.join(os.getcwd(), 'agents'))

print("="*70)
print("🔍 Complete Training Setup Verification (Fixed)")
print("="*70)

# ============================================
# 1. بررسی import paths
# ============================================
print("\n📋 Step 1: Import Path Check")
print(f"   sys.path[0]: {sys.path[0]}")
print(f"   sys.path[1]: {sys.path[1]}")

# ============================================
# 2. محیط
# ============================================
print("\n📋 Step 2: Environment Setup")
from uav_mec_env import UAVMECEnvironment

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
# 3. Agent Setup
# ============================================
print("\n📋 Step 3: Agent Setup")

# بررسی کدام MADDPGAgent import می‌شود
from maddpg_agent import MADDPGAgent
import inspect

agent_file = inspect.getfile(MADDPGAgent)
print(f"   Using MADDPGAgent from: {agent_file}")

sig = inspect.signature(MADDPGAgent.__init__)
params = list(sig.parameters.keys())
print(f"   Parameters: {params[1:6]}...")  # نمایش 5 پارامتر اول (بدون self)

# ============================================
# 4. ساخت Agent با پارامترهای صحیح
# ============================================
print("\n📋 Step 4: Creating Agent")

total_state_dim = state_dim * env.num_uavs
total_action_dim = action_dim * env.num_uavs

print(f"   state_dim: {state_dim}")
print(f"   action_dim: {action_dim}")
print(f"   total_state_dim: {total_state_dim}")
print(f"   total_action_dim: {total_action_dim}")

try:
    # تلاش برای ساخت agent با پارامترهای multi-agent
    agent = MADDPGAgent(
        agent_id=0,
        state_dim=state_dim,
        action_dim=action_dim,
        total_state_dim=total_state_dim,
        total_action_dim=total_action_dim
    )
    print(f"   ✅ Agent created successfully (multi-agent version)")
    agent_type = "multi-agent"
    
except TypeError as e:
    print(f"   ⚠️  Multi-agent version failed: {e}")
    print(f"   Trying single-agent version...")
    
    try:
        # تلاش با پارامترهای single-agent
        agent = MADDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim
        )
        print(f"   ✅ Agent created successfully (single-agent version)")
        agent_type = "single-agent"
    except Exception as e2:
        print(f"   ❌ Both versions failed!")
        print(f"   Error: {e2}")
        exit(1)

# ============================================
# 5. بررسی Actor
# ============================================
print(f"\n📋 Step 5: Actor Network Check")
import torch

print(f"   Actor type: {type(agent.actor).__name__}")

# بررسی layers
if hasattr(agent.actor, 'fc1'):
    fc1_in = agent.actor.fc1.in_features
    fc1_out = agent.actor.fc1.out_features
    print(f"   fc1: {fc1_in} → {fc1_out}")
    
    if fc1_in != state_dim:
        print(f"   ⚠️  WARNING: Actor expects {fc1_in}, but env provides {state_dim}")
    else:
        print(f"   ✅ Actor input matches state_dim")

# تست forward pass
try:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    print(f"\n   Testing forward pass...")
    print(f"   Input shape: {state_tensor.shape}")
    
    with torch.no_grad():
        if hasattr(agent.actor, 'forward'):
            output = agent.actor(state_tensor)
            if isinstance(output, tuple):
                print(f"   ✅ Forward pass OK (tuple output)")
                print(f"   Output shapes: {[o.shape for o in output]}")
            else:
                print(f"   ✅ Forward pass OK")
                print(f"   Output shape: {output.shape}")
        else:
            print(f"   ⚠️  No forward method found")
            
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================
# 6. خلاصه
# ============================================
print("\n" + "="*70)
print("🎯 Verification Summary:")
print(f"   Agent version: {agent_type}")
print(f"   State dimension: {state_dim}")
print(f"   Action dimension: {action_dim}")
print(f"   Agent file: {os.path.basename(agent_file)}")
print("="*70)
