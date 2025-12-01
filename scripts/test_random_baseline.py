"""
Test Random Baseline with UAVMECEnvironment
Official test script for Random Policy baseline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from environments.uav_mec_env import UAVMECEnvironment, UAVConfig, TaskConfig
from algorithms.baselines.random_policy import RandomAgent

def test_random_agent(num_episodes=5, max_steps=100):
    """Test Random Agent performance"""
    
    print("=" * 70)
    print("🧪 Testing Random Agent with UAVMECEnvironment")
    print("=" * 70)
    
    # Create environment
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
    
    # Create agent
    print("\n🤖 Creating Random Agent...")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = RandomAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        action_space=env.action_space
    )
    print(f"✅ Agent created (obs_dim={obs_dim}, action_dim={action_dim})")
    
    # Run episodes
    print(f"\n🎮 Running {num_episodes} test episodes...")
    print("=" * 70)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        print(f"\n📊 Episode {episode + 1}/{num_episodes}")
        
        while not (done or truncated) and steps < max_steps:
            action = agent.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            obs = next_obs
            
            if steps % 20 == 0:
                print(f"   Step {steps}: reward={reward:.2f}, total={episode_reward:.2f}")
        
        episode_rewards.append(episode_reward)
        print(f"✅ Episode finished: steps={steps}, total_reward={episode_reward:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 Summary Statistics:")
    print(f"   Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"   Std Reward:  {np.std(episode_rewards):.2f}")
    print(f"   Min Reward:  {np.min(episode_rewards):.2f}")
    print(f"   Max Reward:  {np.max(episode_rewards):.2f}")
    print("=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)
    
    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'all_rewards': episode_rewards
    }

if __name__ == "__main__":
    results = test_random_agent(num_episodes=5, max_steps=100)
