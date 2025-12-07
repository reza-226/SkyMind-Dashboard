"""
debug_env_info.py - بررسی خروجی محیط واقعی پروژه
"""
import sys
import os

# Find the actual environment being used
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

import numpy as np
import torch

# Try to import the actual environment from your project
try:
    # Option 1: Your custom UAV environment
    from envs.uav_env import UAVEnvironment
    USE_CUSTOM = True
    print("✅ Using custom UAVEnvironment")
except ImportError:
    # Option 2: PettingZoo MPE (based on your ablation study)
    try:
        from pettingzoo.mpe import simple_tag_v3
        USE_CUSTOM = False
        print("✅ Using PettingZoo simple_tag_v3")
    except ImportError:
        print("❌ No environment found!")
        print("Please check:")
        print("1. Is 'envs/uav_env.py' present?")
        print("2. Is PettingZoo installed? (pip install pettingzoo)")
        sys.exit(1)

def debug_environment():
    """Debug the actual environment"""
    
    print("=" * 60)
    print("🔍 Debug Environment Info")
    print("=" * 60)
    
    if USE_CUSTOM:
        # Custom UAV Environment (single agent)
        env = UAVEnvironment()
        obs, info = env.reset()
        
        print(f"\n📋 Initial Info: {info}")
        print(f"📊 Observation Shape: {obs.shape}")
        print(f"🎯 Action Space: {env.action_space}")
        
        # Run 10 steps
        print("\n" + "=" * 60)
        print("🎮 Running 10 Random Steps...")
        print("=" * 60)
        
        total_reward = 0
        for step in range(10):
            action = np.concatenate([
                np.random.randn(5),
                np.random.uniform(-1, 1, 6)
            ])
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"\n--- Step {step+1} ---")
            print(f"  Reward: {reward:.4f}")
            print(f"  Done: {done}, Truncated: {truncated}")
            print(f"  Info: {info}")
            
            if done or truncated:
                break
                
    else:
        # PettingZoo Multi-Agent Environment
        env = simple_tag_v3.env()
        env.reset()
        
        print(f"\n📊 Agents: {env.agents}")
        print(f"📊 Number of agents: {len(env.agents)}")
        
        # Run one episode
        print("\n" + "=" * 60)
        print("🎮 Running One Episode...")
        print("=" * 60)
        
        episode_rewards = {agent: 0 for agent in env.agents}
        step_count = 0
        
        for agent in env.agent_iter(max_iter=50):
            observation, reward, termination, truncation, info = env.last()
            
            if step_count < 5:  # Print first 5 steps only
                print(f"\n--- Step {step_count}, Agent: {agent} ---")
                print(f"  Obs shape: {observation.shape if hasattr(observation, 'shape') else len(observation)}")
                print(f"  Reward: {reward:.4f}")
                print(f"  Done: {termination or truncation}")
                print(f"  Info: {info}")
            
            episode_rewards[agent] += reward
            
            if termination or truncation:
                action = None
            else:
                # Random action
                action = env.action_space(agent).sample()
            
            env.step(action)
            step_count += 1
        
        print("\n" + "=" * 60)
        print("📈 Episode Summary:")
        for agent, total in episode_rewards.items():
            print(f"  {agent}: {total:.4f}")
        print("=" * 60)

if __name__ == "__main__":
    debug_environment()
