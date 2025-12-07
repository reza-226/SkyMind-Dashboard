# test_networks.py
import torch
from models.actor_critic import ActorNetwork, CriticNetwork, ActionDecoder

# Test Actor
actor = ActorNetwork(state_dim=537, offload_dim=5, continuous_dim=6)
state = torch.randn(32, 537)
offload_logits, cont = actor(state)
print(f"✅ Actor: {offload_logits.shape}, {cont.shape}")

# Test Critic
critic = CriticNetwork(state_dim=537, action_dim=11)
action = torch.randn(32, 11)
q_value = critic(state, action)
print(f"✅ Critic: {q_value.shape}")

# Test Decoder
decoder = ActionDecoder()
actions = decoder.decode(offload_logits, cont)
print(f"✅ Decoder: {len(actions)} actions")
print(f"Sample action: {actions[0]}")
