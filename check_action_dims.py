"""
Check action dimensions for matching environments
"""

import warnings
warnings.filterwarnings('ignore')

from pettingzoo.mpe import simple_adversary_v3, simple_tag_v3

print("=" * 60)
print("🔍 CHECKING ACTION DIMENSIONS")
print("=" * 60)

# Test simple_adversary_v3 (N=3)
print("\n📌 simple_adversary_v3 (N=3):")
env = simple_adversary_v3.parallel_env(N=3, continuous_actions=True)
obs, _ = env.reset()
for agent in env.agents:
    action_space = env.action_space(agent)
    obs_dim = obs[agent].shape[0]
    print(f"   {agent}: obs_dim={obs_dim}, action_space={action_space}")
env.close()

# Test simple_tag_v3 (N=2)
print("\n📌 simple_tag_v3 (num_good=2, num_adversaries=1):")
env = simple_tag_v3.parallel_env(num_good=2, num_adversaries=1, continuous_actions=True)
obs, _ = env.reset()
for agent in env.agents:
    action_space = env.action_space(agent)
    obs_dim = obs[agent].shape[0]
    print(f"   {agent}: obs_dim={obs_dim}, action_space={action_space}")
env.close()

print("\n" + "=" * 60)
print("🎯 YOUR MODEL: obs_dim=14, action_dim=5, num_agents=2")
print("=" * 60)
