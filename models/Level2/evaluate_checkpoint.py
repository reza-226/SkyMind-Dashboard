# evaluate_checkpoint.py
import sys
import os
from pathlib import Path

# ✅ Fix: Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

try:
    from uav_offloading.envs.simple_uav_env import UAVOffloadingEnv
    print("✅ Module imported successfully!")
except ModuleNotFoundError as e:
    print(f"❌ Import error: {e}")
    print(f"Project root: {project_root}")
    print(f"sys.path[0]: {sys.path[0]}")
    sys.exit(1)

def evaluate_checkpoint(checkpoint_path, num_episodes=50):
    """ارزیابی واقعی عملکرد Checkpoint در محیط"""
    
    print(f"🎮 Evaluating checkpoint: {checkpoint_path}")
    print(f"📊 Running {num_episodes} test episodes...\n")
    
    # بارگذاری مدل‌ها
    checkpoint_path = Path(checkpoint_path)
    agent_0 = torch.load(checkpoint_path / "agent_0.pth", map_location='cpu')
    agent_1 = torch.load(checkpoint_path / "agent_1.pth", map_location='cpu')
    
    # ساخت محیط (Level 2)
    env = UAVOffloadingEnv(
        num_uavs=2,
        num_tasks=15,
        map_size=200,
        max_steps=100
    )
    
    # ساخت Policy Network (مشابه Actor)
    class PolicyNet(torch.nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(17, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.LayerNorm(128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.LayerNorm(64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 5),
                torch.nn.Tanh()
            )
            self.load_state_dict(state_dict)
            self.eval()
        
        def forward(self, x):
            return self.net(x)
    
    policy_0 = PolicyNet(agent_0)
    policy_1 = PolicyNet(agent_1)
    
    # تست
    rewards = []
    action_warnings = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 100:
            with torch.no_grad():
                obs_0 = torch.FloatTensor(obs[0]).unsqueeze(0)
                obs_1 = torch.FloatTensor(obs[1]).unsqueeze(0)
                
                action_0 = policy_0(obs_0).squeeze(0).numpy()
                action_1 = policy_1(obs_1).squeeze(0).numpy()
                
                # Tanh -> [0, 1]
                action_0 = (action_0 + 1) / 2
                action_1 = (action_1 + 1) / 2
                
                if not (0 <= action_0.min() and action_0.max() <= 1):
                    action_warnings += 1
                if not (0 <= action_1.min() and action_1.max() <= 1):
                    action_warnings += 1
                
                actions = [action_0, action_1]
            
            obs, rewards_step, done, info = env.step(actions)
            episode_reward += sum(rewards_step)
            step += 1
        
        rewards.append(episode_reward)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: Mean Reward = {np.mean(rewards[-10:]):.2f}")
    
    # نتایج
    print(f"\n{'='*60}")
    print(f"📊 **Evaluation Results:**")
    print(f"{'='*60}")
    print(f"  Mean Reward:   {np.mean(rewards):>10.2f}")
    print(f"  Std Reward:    {np.std(rewards):>10.2f}")
    print(f"  Min Reward:    {np.min(rewards):>10.2f}")
    print(f"  Max Reward:    {np.max(rewards):>10.2f}")
    print(f"  Action Warnings: {action_warnings}")
    print(f"{'='*60}\n")
    
    mean_reward = np.mean(rewards)
    
    if mean_reward > -50:
        print(f"✅ **وضعیت: عالی!**")
    elif mean_reward > -100:
        print(f"⚠️ **وضعیت: متوسط.**")
    else:
        print(f"❌ **وضعیت: بد!**")
    
    return mean_reward

if __name__ == "__main__":
    evaluate_checkpoint("models/level2/checkpoint_7000", num_episodes=50)
