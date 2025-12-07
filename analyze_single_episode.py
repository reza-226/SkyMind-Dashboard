# analyze_single_episode.py
"""تحلیل دقیق یک episode"""
import torch
import numpy as np
from pathlib import Path
import json
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from environments.uav_mec_env import UAVMECEnvironment
from models.actor_critic.maddpg_agent import MADDPGAgent

def analyze_episode(model_path):
    """تحلیل یک episode با جزئیات"""
    
    # بارگذاری config و model
    model_dir = Path(model_path).parent
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    env_config = config['env_config']
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    
    env = UAVMECEnvironment(**env_config)
    
    agent = MADDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=512,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.01
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor.eval()
    
    # اجرای یک episode
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    print(f"\n{'='*70}")
    print(f"📊 Detailed Episode Analysis")
    print(f"{'='*70}\n")
    
    episode_data = {
        'steps': [],
        'states': [],
        'actions': [],
        'rewards': [],
        'info': []
    }
    
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < 50:
        # Select action
        with torch.no_grad():
            action = agent.select_action(state, noise=0.0)
        
        # Execute
        result = env.step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
        else:
            break
        
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        
        # ذخیره داده‌ها
        episode_data['steps'].append(step)
        episode_data['states'].append(state.copy())
        episode_data['actions'].append(action.copy())
        episode_data['rewards'].append(reward)
        episode_data['info'].append(info)
        
        # نمایش step
        print(f"Step {step:2d} │ Reward: {reward:8.2f} │ Total: {total_reward:9.2f}")
        
        # نمایش جزئیات info (اگر موجود باشد)
        if isinstance(info, dict):
            interesting_keys = ['penalty', 'success', 'collision', 'energy', 'latency']
            info_str = ' │ '.join([f"{k}: {info.get(k, 'N/A')}" for k in interesting_keys if k in info])
            if info_str:
                print(f"         │ {info_str}")
        
        total_reward += reward
        state = next_state
        step += 1
    
    print(f"\n{'='*70}")
    print(f"📈 Episode Summary")
    print(f"{'='*70}")
    print(f"Total Steps:  {step}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Mean Reward:  {total_reward/step:.2f}")
    print(f"Max Reward:   {max(episode_data['rewards']):.2f}")
    print(f"Min Reward:   {min(episode_data['rewards']):.2f}")
    print(f"{'='*70}\n")
    
    return episode_data

if __name__ == "__main__":
    model_path = 'results/4layer_3level/level_1/best_model.pth'
    data = analyze_episode(model_path)
