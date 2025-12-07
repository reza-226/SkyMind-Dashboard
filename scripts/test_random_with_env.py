"""
Test Random Baseline with UAVMECEnvironment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from environments.uav_mec_env import UAVMECEnvironment, UAVConfig, TaskConfig
from algorithms.baselines.random_policy.random_agent import RandomAgent

print("=" * 70)
print("🧪 Testing Random Agent with UAVMECEnvironment")
print("=" * 70)

# 1. Create environment
print("\n📦 Creating environment...")
env = UAVMECEnvironment(
    uav_config=UAVConfig(num_uavs=2),
    task_config=TaskConfig(num_tasks=5),
    grid_size=(500.0, 500.0),
    num_obstacles=3,
    difficulty='easy'
)

print(f"✅ Environment created!")
print(f"   Observation space: {env.observation_space.shape}")
print(f"   Action space: {env.action_space.shape}")

# 2. Create Random Agent
print("\n🤖 Creating Random Agent...")
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = RandomAgent(
    observation_dim=obs_dim,
    action_dim=action_dim,
    action_space=env.action_space
)
print(f"✅ Agent created with obs_dim={obs_dim}, action_dim={action_dim}")

# 3. Run test episodes
print("\n🎮 Running test episodes...")
print("=" * 70)

num_episodes = 5
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    steps = 0
    
    print(f"\n📊 Episode {episode + 1}/{num_episodes}")
    print(f"   Initial obs shape: {obs.shape}")
    
    while not (done or truncated) and steps < 50:
        # Get action from agent
        action = agent.get_action(obs)
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        steps += 1
        
        obs = next_obs
        
        if steps % 10 == 0:
            print(f"   Step {steps}: reward={reward:.2f}, total={episode_reward:.2f}")
    
    print(f"✅ Episode finished: steps={steps}, total_reward={episode_reward:.2f}")
    print(f"   Info: {info}")

print("\n" + "=" * 70)
print("✅ All tests completed successfully!")
print("=" * 70)
