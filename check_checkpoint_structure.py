"""
check_checkpoint_structure.py
بررسی ساختار checkpoint برای شناسایی کلیدهای موجود
"""

import torch
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """بررسی دقیق ساختار checkpoint"""
    
    print(f"\n{'='*70}")
    print(f"🔍 Inspecting Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("📋 Top-level Keys:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            value_type = type(value).__name__
            
            if isinstance(value, dict):
                print(f"   • {key:20s} → dict with {len(value)} keys")
                for subkey in list(value.keys())[:5]:
                    print(f"      ├─ {subkey}")
                if len(value) > 5:
                    print(f"      └─ ... and {len(value)-5} more")
                    
            elif isinstance(value, list):
                print(f"   • {key:20s} → list with {len(value)} elements")
                if len(value) > 0:
                    print(f"      └─ First element type: {type(value[0]).__name__}")
                    
            elif isinstance(value, torch.Tensor):
                print(f"   • {key:20s} → Tensor {tuple(value.shape)}")
                
            else:
                print(f"   • {key:20s} → {value_type}: {value}")
        
        # Check for actor networks
        print(f"\n{'='*70}")
        print("🤖 Looking for Actor Networks...")
        print(f"{'='*70}\n")
        
        # Common patterns
        possible_keys = [
            'actors', 'actor', 'actor_state_dict', 'actor_networks',
            'agent_0', 'agent_1', 'model', 'models', 'state_dict'
        ]
        
        for key in possible_keys:
            if key in checkpoint:
                print(f"   ✅ Found: '{key}'")
                value = checkpoint[key]
                
                if isinstance(value, list):
                    print(f"      └─ List with {len(value)} actors")
                    if len(value) > 0 and isinstance(value[0], dict):
                        print(f"         First actor keys: {list(value[0].keys())[:5]}")
                        
                elif isinstance(value, dict):
                    print(f"      └─ Dict with keys: {list(value.keys())[:10]}")
        
        # Check state_dim
        print(f"\n{'='*70}")
        print("📊 Looking for State Dimension Info...")
        print(f"{'='*70}\n")
        
        dim_keys = ['state_dim', 'obs_dim', 'observation_dim', 'input_dim']
        for key in dim_keys:
            if key in checkpoint:
                print(f"   ✅ Found: '{key}' = {checkpoint[key]}")
        
        # Try to infer from actor weights
        print(f"\n{'='*70}")
        print("🔍 Inferring Architecture from Weights...")
        print(f"{'='*70}\n")
        
        actor_dict = None
        if 'actors' in checkpoint and isinstance(checkpoint['actors'], list):
            actor_dict = checkpoint['actors'][0]
        elif 'actor' in checkpoint:
            actor_dict = checkpoint['actor']
        elif 'state_dict' in checkpoint:
            actor_dict = checkpoint['state_dict']
        
        if actor_dict and isinstance(actor_dict, dict):
            print("   Actor Layer Shapes:")
            for key, value in actor_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"      • {key:30s} → {tuple(value.shape)}")
            
            # Infer dimensions
            if 'fc1.weight' in actor_dict:
                state_dim = actor_dict['fc1.weight'].shape[1]
                hidden_dim = actor_dict['fc1.weight'].shape[0]
                print(f"\n   📐 Inferred Dimensions:")
                print(f"      • State Dim:  {state_dim}")
                print(f"      • Hidden Dim: {hidden_dim}")
            
            if 'offload_head.weight' in actor_dict:
                offload_dim = actor_dict['offload_head.weight'].shape[0]
                print(f"      • Offload Dim: {offload_dim}")
            
            if 'continuous_head.weight' in actor_dict:
                cont_dim = actor_dict['continuous_head.weight'].shape[0]
                print(f"      • Continuous Dim: {cont_dim}")
        
        print(f"\n{'='*70}")
        print("✅ Inspection Complete!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    checkpoint_path = "checkpoints/maddpg/best_model.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
    else:
        inspect_checkpoint(checkpoint_path)
