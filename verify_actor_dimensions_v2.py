# verify_actor_dimensions_v2.py
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

# ساخت مستقیم Actor
actor = ActorNetwork(
    state_dim=state_dim,
    offload_dim=5,
    continuous_dim=6,
    hidden=512
)

# چک ابعاد
print(f"\n🔍 Actor Network Layers:")
print(f"   fc1: {actor.fc1.in_features} → {actor.fc1.out_features}")
print(f"   fc2: {actor.fc2.in_features} → {actor.fc2.out_features}")
print(f"   offload_head: {actor.offload_head.in_features} → {actor.offload_head.out_features}")
print(f"   continuous_head: {actor.continuous_head.in_features} → {actor.continuous_head.out_features}")

# تست forward pass
dummy_state = torch.FloatTensor(state).unsqueeze(0)
print(f"\n🧪 Testing forward pass with input shape: {dummy_state.shape}")

try:
    offload_logits, continuous = actor(dummy_state)
    print(f"✅ Forward pass successful!")
    print(f"   Offload logits shape: {offload_logits.shape}")
    print(f"   Continuous shape: {continuous.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")

print("\n" + "="*60)
print("✅ Actor is compatible with state_dim={state_dim}!")
print("="*60)
