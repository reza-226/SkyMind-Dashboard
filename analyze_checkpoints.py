"""
Analyze Training Checkpoints - Check Environment Configuration
"""

import torch
from pathlib import Path
import json

def analyze_checkpoint(checkpoint_path):
    """Extract all info from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            'file': checkpoint_path.name,
            'keys': list(checkpoint.keys())
        }
        
        # Extract network dimensions
        if 'actor' in checkpoint:
            actor_state = checkpoint['actor']
            
            # First layer
            first_weight = actor_state['network.0.weight']
            hidden_dim, obs_dim = first_weight.shape
            
            # Last layer
            last_weight = actor_state['network.4.weight']
            action_dim = last_weight.shape[0]
            
            info['obs_dim'] = obs_dim
            info['action_dim'] = action_dim
            info['hidden_dim'] = hidden_dim
            
            # All layer shapes
            info['layers'] = {}
            for key, value in actor_state.items():
                if 'weight' in key:
                    info['layers'][key] = list(value.shape)
        
        # Check for config
        if 'config' in checkpoint:
            info['saved_config'] = checkpoint['config']
        
        # Check for episode/step info
        if 'episode' in checkpoint:
            info['episode'] = checkpoint['episode']
        if 'step' in checkpoint:
            info['step'] = checkpoint['step']
            
        return info
        
    except Exception as e:
        return {'file': checkpoint_path.name, 'error': str(e)}

def main():
    print("=" * 80)
    print("🔍 CHECKPOINT ANALYSIS")
    print("=" * 80)
    
    model_dir = Path("models/level3_complex/final")
    
    for ckpt_path in sorted(model_dir.glob("*.pth")):
        info = analyze_checkpoint(ckpt_path)
        
        print(f"\n📦 {info['file']}")
        print("-" * 80)
        
        if 'error' in info:
            print(f"   ❌ Error: {info['error']}")
            continue
        
        print(f"   Checkpoint Keys: {info['keys']}")
        
        if 'obs_dim' in info:
            print(f"\n   Network Dimensions:")
            print(f"      obs_dim     = {info['obs_dim']}")
            print(f"      action_dim  = {info['action_dim']}")
            print(f"      hidden_dim  = {info['hidden_dim']}")
            
            # Calculate required N
            obs = info['obs_dim']
            # Formula: obs = 4 + 4*N
            if (obs - 4) % 4 == 0:
                n = (obs - 4) // 4
                print(f"      Required N  = {n} agents")
            else:
                print(f"      ⚠️  obs_dim={obs} doesn't match formula (4 + 4*N)")
        
        if 'layers' in info:
            print(f"\n   Layer Shapes:")
            for layer_name, shape in info['layers'].items():
                print(f"      {layer_name}: {shape}")
        
        if 'saved_config' in info:
            print(f"\n   Saved Config: {info['saved_config']}")
        
        if 'episode' in info:
            print(f"\n   Training Info:")
            print(f"      Episode: {info['episode']}")
            if 'step' in info:
                print(f"      Step: {info['step']}")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)
    print("If obs_dim doesn't match standard N values (2, 3, 4...):")
    print("1. Check training script configuration")
    print("2. Models might be trained on custom environment")
    print("3. Need to recreate exact training environment")
    print("=" * 80)

if __name__ == '__main__':
    main()
