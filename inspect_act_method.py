"""
inspect_act_method.py
بررسی دقیق متد act() در MADDPG_Agent
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from agents.agent_maddpg_multi import MADDPG_Agent

print("=" * 60)
print("Testing MADDPG_Agent.act() method")
print("=" * 60)

# ساخت agent با پارامترهای واقعی
state_dim = 38
action_dim = 4
n_agents = 3

agent = MADDPG_Agent(
    state_dim=state_dim,
    action_dim=action_dim,
    n_agents=n_agents,
    lr=1e-4,
    gamma=0.95
)

print(f"\n📋 Agent Configuration:")
print(f"   state_dim: {state_dim}")
print(f"   action_dim: {action_dim}")
print(f"   n_agents: {n_agents}")

# تست با state واقعی
test_state = np.random.randn(state_dim)

print(f"\n🧪 Test Input:")
print(f"   State shape: {test_state.shape}")
print(f"   State: {test_state[:5]}... (showing first 5)")

# فراخوانی act
print(f"\n🚀 Calling agent.act(state)...")
try:
    actions = agent.act(test_state)
    
    print(f"\n✅ Output:")
    print(f"   Type: {type(actions)}")
    
    if isinstance(actions, np.ndarray):
        print(f"   Shape: {actions.shape}")
        print(f"   Size: {actions.size}")
        print(f"   Actions: {actions}")
    elif isinstance(actions, list):
        print(f"   Length: {len(actions)}")
        print(f"   Actions: {actions}")
    else:
        print(f"   Value: {actions}")
    
    # بررسی اگر باید reshape شود
    if isinstance(actions, np.ndarray):
        if actions.size == n_agents * action_dim:
            reshaped = actions.reshape(n_agents, action_dim)
            print(f"\n🔄 Can be reshaped to ({n_agents}, {action_dim}):")
            print(f"   {reshaped}")
        else:
            print(f"\n⚠️  Size mismatch! Expected {n_agents * action_dim}, got {actions.size}")
            
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

# بررسی ساختار داخلی agent
print(f"\n🔍 Agent Structure:")
if hasattr(agent, 'actors'):
    print(f"   Has 'actors' attribute")
    print(f"   Number of actors: {len(agent.actors)}")
    
    # تست تک‌تک actors
    print(f"\n🧪 Testing individual actors:")
    for i in range(min(3, len(agent.actors))):
        try:
            # فرض: هر actor یک state می‌گیرد
            single_action = agent.actors[i](
                np.random.randn(state_dim)
            )
            print(f"   Actor {i}: output shape = {single_action.shape if hasattr(single_action, 'shape') else 'N/A'}")
        except Exception as e:
            print(f"   Actor {i}: Error - {e}")

print("\n" + "=" * 60)
