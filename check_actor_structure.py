# check_actor_structure.py
from models.actor_critic.maddpg_agent import MADDPGAgent

agent = MADDPGAgent(state_dim=71, action_dim=7)

print("📋 Actor attributes:")
for attr in dir(agent.actor):
    if not attr.startswith('_'):
        print(f"   - {attr}")

print("\n🔍 Actor layers:")
for name, module in agent.actor.named_modules():
    if name:  # skip root
        print(f"   {name}: {module}")
