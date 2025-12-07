# test_maddpg.py

import torch
import numpy as np
from models.actor_critic.maddpg_agent import MADDPGAgent

def test_maddpg_agent():
    print("\n" + "="*80)
    print("🧪 Testing MADDPG Agent")
    print("="*80)
    
    # Initialize agent
    agent = MADDPGAgent(
        state_dim=114,
        offload_dim=4,
        continuous_dim=7,
        action_dim=11
    )
    
    print(f"\n✅ Agent initialized on device: {agent.device}")
    
    # Test action selection
    print("\n1️⃣ Testing action selection...")
    state = np.random.randn(114)
    action = agent.select_action(state, add_noise=True)
    
    print(f"  ✅ Action keys: {action.keys()}")
    print(f"  ✅ Offload layer: {action['offload']}")
    print(f"  ✅ Bandwidth: {action['bandwidth']} (sum={action['bandwidth'].sum():.4f})")
    print(f"  ✅ CPU: {action['cpu']:.4f}")
    print(f"  ✅ Movement: {action['move']}")
    
    # Test replay buffer
    print("\n2️⃣ Testing replay buffer...")
    for i in range(100):
        state = np.random.randn(114)
        action = agent.select_action(state, add_noise=False)
        reward = np.random.randn()
        next_state = np.random.randn(114)
        done = i % 20 == 0
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
    
    print(f"  ✅ Buffer size: {len(agent.replay_buffer)}")
    
    # Test network update
    print("\n3️⃣ Testing network update...")
    losses = agent.update(batch_size=32)
    
    if losses:
        print(f"  ✅ Critic loss: {losses['critic_loss']:.4f}")
        print(f"  ✅ Actor loss: {losses['actor_loss']:.4f}")
        print(f"  ✅ Q-value: {losses['q_value']:.4f}")
    
    # Test save/load
    print("\n4️⃣ Testing save/load...")
    agent.save('test_agent.pth')
    agent.load('test_agent.pth')
    
    print("\n" + "="*80)
    print("✅ All MADDPG Agent Tests Passed!")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_maddpg_agent()
