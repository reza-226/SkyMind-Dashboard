# check_action_distribution.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from environments.uav_mec_env import UAVMECEnvironment
from models.actor_critic.maddpg_agent import MADDPGAgent

def analyze_actions(model_path, num_samples=1000):
    """تحلیل توزیع actions مدل"""
    
    # Load config & model
    model_dir = Path(model_path).parent
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    
    agent = MADDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=512,
        lr_actor=1e-4,
        lr_critic=1e-3
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor.eval()
    
    # جمع‌آوری actions از states تصادفی
    actions = []
    
    for _ in range(num_samples):
        # State تصادفی
        state = np.random.randn(state_dim)
        
        with torch.no_grad():
            action = agent.select_action(state, noise=0.0)
        
        actions.append(action)
    
    actions = np.array(actions)
    
    # تحلیل
    print(f"\n{'='*70}")
    print(f"🎯 Action Distribution Analysis ({num_samples} samples)")
    print(f"{'='*70}\n")
    
    for i in range(action_dim):
        print(f"Action dim {i}:")
        print(f"  Mean:  {actions[:, i].mean():7.3f}")
        print(f"  Std:   {actions[:, i].std():7.3f}")
        print(f"  Min:   {actions[:, i].min():7.3f}")
        print(f"  Max:   {actions[:, i].max():7.3f}")
        print()
    
    # رسم نمودار
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(action_dim):
        axes[i].hist(actions[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Action Dim {i}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('action_distribution.png', dpi=150)
    plt.show()
    
    print(f"✅ Analysis saved to 'action_distribution.png'")

if __name__ == "__main__":
    analyze_actions('results/4layer_3level/level_1/best_model.pth')
