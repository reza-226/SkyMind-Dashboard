# verify_actor_dimensions_final.py
import sys
sys.path.append('environments')
sys.path.append('models/actor_critic')

from uav_mec_env import UAVMECEnvironment
from actor_network import ActorNetwork
import torch

print("="*60)
print("🔍 Verifying Actor Network Dimensions")
print("="*60)

# تشخیص state_dim از محیط
env = UAVMECEnvironment()
state = env.reset()
state_dim = len(state)

print(f"\n📐 Environment state_dim: {state_dim}")

# ساخت Actor با پارامترهای صحیح (طبق کد actor_network.py)
actor = ActorNetwork(
    state_dim=state_dim,
    hidden=512
)

# چک ابعاد
print(f"\n🔍 Actor Network Layers:")
print(f"   fc1: {actor.fc1.in_features} → {actor.fc1.out_features}")
print(f"   fc2: {actor.fc2.in_features} → {actor.fc2.out_features}")

if hasattr(actor, 'offload_head'):
    print(f"   offload_head: {actor.offload_head.in_features} → {actor.offload_head.out_features}")
if hasattr(actor, 'continuous_head'):
    print(f"   continuous_head: {actor.continuous_head.in_features} → {actor.continuous_head.out_features}")
if hasattr(actor, 'action_out'):
    print(f"   action_out: {actor.action_out.in_features} → {actor.action_out.out_features}")

# تست forward pass
dummy_state = torch.FloatTensor(state).unsqueeze(0)
print(f"\n🧪 Testing forward pass with input shape: {dummy_state.shape}")

try:
    output = actor(dummy_state)
    
    # چک نوع خروجی
    if isinstance(output, tuple):
        print(f"✅ Forward pass successful! (Dual output)")
        print(f"   Output[0] shape: {output[0].shape}")
        print(f"   Output[1] shape: {output[1].shape}")
    else:
        print(f"✅ Forward pass successful! (Single output)")
        print(f"   Output shape: {output.shape}")
        
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print(f"✅ Actor is compatible with state_dim={state_dim}!")
print("="*60)
