"""
Find the correct environment for trained models
Models: agent_0 (obs_dim=6), adversary_0 (obs_dim=8)
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pettingzoo.mpe import (
    simple_v3,
    simple_spread_v3,
    simple_tag_v3,
    simple_adversary_v3,
    simple_push_v3,
    simple_reference_v3,
    simple_speaker_listener_v4,
)

def test_env(name, env_fn, **kwargs):
    """Test an environment and print observation dimensions"""
    try:
        env = env_fn(**kwargs)
        env.reset()
        
        obs_dims = {}
        for agent in env.agents:
            obs_space = env.observation_space(agent)
            obs_dims[agent] = obs_space.shape[0]
        
        env.close()
        
        print(f"\n✅ {name}")
        print(f"   Agents: {list(obs_dims.keys())}")
        print(f"   Observation dims: {list(obs_dims.values())}")
        
        # Check if matches our model
        dims = list(obs_dims.values())
        if 6 in dims or 8 in dims:
            print(f"   🎯 POSSIBLE MATCH!")
        
        return obs_dims
        
    except Exception as e:
        print(f"\n❌ {name}: {e}")
        return None

print("=" * 80)
print("🔬 SEARCHING FOR ENVIRONMENT WITH obs_dim=6 (agent) and obs_dim=8 (adversary)")
print("=" * 80)

# Test various environments with different parameters
test_env("simple_v3", simple_v3.parallel_env)

test_env("simple_spread_v3 (N=2)", simple_spread_v3.parallel_env, N=2)
test_env("simple_spread_v3 (N=3)", simple_spread_v3.parallel_env, N=3)

test_env("simple_tag_v3 (good=1, adv=1)", simple_tag_v3.parallel_env, 
         num_good=1, num_adversaries=1, num_obstacles=0, continuous_actions=True)
test_env("simple_tag_v3 (good=1, adv=1, obs=1)", simple_tag_v3.parallel_env, 
         num_good=1, num_adversaries=1, num_obstacles=1, continuous_actions=True)
test_env("simple_tag_v3 (good=1, adv=1, obs=2)", simple_tag_v3.parallel_env, 
         num_good=1, num_adversaries=1, num_obstacles=2, continuous_actions=True)

test_env("simple_adversary_v3 (N=2)", simple_adversary_v3.parallel_env, N=2)
test_env("simple_adversary_v3 (N=3)", simple_adversary_v3.parallel_env, N=3)

test_env("simple_push_v3", simple_push_v3.parallel_env)

test_env("simple_reference_v3", simple_reference_v3.parallel_env)

print("\n" + "=" * 80)
print("💡 Look for environment where agent=6 and adversary=8")
print("=" * 80)
