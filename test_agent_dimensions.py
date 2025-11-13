# test_agent_dimensions.py
"""
تست ابعاد Agent برای تشخیص مشکل
"""

import torch
import numpy as np
from agents.agent_maddpg_multi import MADDPG_Agent

def test_agent_dimensions():
    print("="*70)
    print("🔍 تست ابعاد Agent")
    print("="*70)
    
    # پارامترها
    state_dim = 38
    action_dim = 4
    n_agents = 3
    batch_size = 128
    
    print(f"\n📋 پارامترها:")
    print(f"   state_dim: {state_dim}")
    print(f"   action_dim: {action_dim}")
    print(f"   n_agents: {n_agents}")
    print(f"   batch_size: {batch_size}")
    
    # ساخت agents
    print(f"\n🔧 ساخت {n_agents} agents...")
    agents = []
    for i in range(n_agents):
        try:
            agent = MADDPG_Agent(
                state_dim=state_dim,
                action_dim=action_dim,
                n_agents=n_agents,
                lr=1e-4,
                gamma=0.99
            )
            agents.append(agent)
            print(f"   ✓ Agent {i} ساخته شد")
            
            # بررسی ساختار
            if hasattr(agent, 'actor'):
                print(f"     Actor input dim: {agent.actor.fc1.in_features if hasattr(agent.actor, 'fc1') else 'Unknown'}")
            if hasattr(agent, 'critic'):
                print(f"     Critic input dim: {agent.critic.fc1.in_features if hasattr(agent.critic, 'fc1') else 'Unknown'}")
        except Exception as e:
            print(f"   ✗ خطا در ساخت Agent {i}: {e}")
            return
    
    # تست ابعاد
    print(f"\n📊 تست ابعاد:")
    states = torch.randn(batch_size, state_dim)
    print(f"   States shape: {states.shape}")
    
    # تست Actor
    print(f"\n🎭 تست Actor outputs:")
    all_actions = []
    for i, agent in enumerate(agents):
        try:
            action = agent.target_actor(states)
            all_actions.append(action)
            print(f"   Agent {i} action shape: {action.shape}")
        except Exception as e:
            print(f"   ✗ خطا در Agent {i} actor: {e}")
            return
    
    # Concatenate actions
    try:
        concatenated_actions = torch.cat(all_actions, dim=1)
        print(f"\n🔗 Concatenated actions:")
        print(f"   Shape: {concatenated_actions.shape}")
        print(f"   Expected: ({batch_size}, {n_agents * action_dim})")
        
        if concatenated_actions.shape[1] != n_agents * action_dim:
            print(f"   ❌ MISMATCH! Got {concatenated_actions.shape[1]}, expected {n_agents * action_dim}")
        else:
            print(f"   ✓ Correct!")
    except Exception as e:
        print(f"   ✗ خطا در concatenation: {e}")
        return
    
    # تست Critic input
    try:
        critic_input = torch.cat([states, concatenated_actions], dim=1)
        print(f"\n🎯 Critic input:")
        print(f"   Shape: {critic_input.shape}")
        print(f"   Expected: ({batch_size}, {state_dim + n_agents * action_dim})")
        
        expected_dim = state_dim + n_agents * action_dim
        if critic_input.shape[1] != expected_dim:
            print(f"   ❌ MISMATCH! Got {critic_input.shape[1]}, expected {expected_dim}")
        else:
            print(f"   ✓ Correct!")
    except Exception as e:
        print(f"   ✗ خطا در critic input: {e}")
        return
    
    # تست Critic forward pass
    print(f"\n🔄 تست Critic forward pass:")
    try:
        q_value = agents[0].critic(states, concatenated_actions)
        print(f"   Q-value shape: {q_value.shape}")
        print(f"   Expected: ({batch_size}, 1)")
        print(f"   ✓ Critic works!")
    except Exception as e:
        print(f"   ✗ خطا در Critic forward: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("✅ همه تست‌ها موفق بود!")
    print("="*70)

if __name__ == "__main__":
    test_agent_dimensions()
