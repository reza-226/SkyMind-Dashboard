# verify_actor_dimensions.py
import sys
sys.path.append('environments')
sys.path.append('models/actor_critic')

from uav_mec_env import UAVMECEnvironment
from maddpg_agent import MADDPGAgent

print("="*60)
print("🔍 Verifying Actor Network Dimensions")
print("="*60)

# تشخیص state_dim از محیط
env = UAVMECEnvironment()
state = env.reset()
state_dim = len(state)

print(f"\n📐 Environment state_dim: {state_dim}")

# ساخت agent
agent = MADDPGAgent(
    state_dim=state_dim,
    action_dim=7,
    hidden_dim=512
)

# چک ابعاد Actor
print(f"\n🔍 Actor Network Layers:")
if hasattr(agent.actor, 'fc1'):
    print(f"   fc1: {agent.actor.fc1.in_features} → {agent.actor.fc1.out_features}")
if hasattr(agent.actor, 'fc2'):
    print(f"   fc2: {agent.actor.fc2.in_features} → {agent.actor.fc2.out_features}")
if hasattr(agent.actor, 'action_out'):
    print(f"   action_out: {agent.actor.action_out.in_features} → {agent.actor.action_out.out_features}")

# تست forward pass
import torch
dummy_state = torch.FloatTensor(state).unsqueeze(0)
print(f"\n🧪 Testing forward pass with shape: {dummy_state.shape}")

try:
    output = agent.actor(dummy_state)
    print(f"✅ Forward pass successful! Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")

print("\n" + "="*60)
print("✅ Verification Complete!")
print("="*60)
