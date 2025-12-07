# test_checkpoint.py
import torch
import numpy as np
from pathlib import Path
import json

def test_checkpoint(checkpoint_path, num_episodes=100):
    """تست عملکرد یک Checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    print(f"🔍 Testing checkpoint: {checkpoint_path}")
    
    # بررسی وجود پوشه
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint path not found: {checkpoint_path}")
        return False
    
    # بررسی فایل‌های موجود
    print(f"\n📁 Available files:")
    pth_files = list(checkpoint_path.glob("*.pth"))
    if not pth_files:
        print(f"❌ No .pth files found in {checkpoint_path}")
        return False
    
    for file in pth_files:
        size = file.stat().st_size / 1024  # KB
        print(f"  ✅ {file.name}: {size:.1f} KB")
    
    # بارگذاری مدل‌ها
    try:
        print(f"\n🔄 Loading models...")
        agent_0 = torch.load(checkpoint_path / "agent_0.pth", map_location='cpu')
        agent_1 = torch.load(checkpoint_path / "agent_1.pth", map_location='cpu')
        adversary = torch.load(checkpoint_path / "adversary_0.pth", map_location='cpu')
        critic = torch.load(checkpoint_path / "critic.pth", map_location='cpu')
        print(f"  ✅ All models loaded successfully!")
    except Exception as e:
        print(f"  ❌ Error loading models: {e}")
        return False
    
    # بررسی وزن‌های مدل
    print(f"\n🔢 Model Statistics:")
    
    for name, model_dict in [("Agent 0", agent_0), 
                              ("Agent 1", agent_1), 
                              ("Adversary", adversary)]:
        if isinstance(model_dict, dict):
            print(f"\n  📊 {name}:")
            for key in model_dict.keys():
                print(f"    - Key: {key}")
                if isinstance(model_dict[key], torch.nn.Module):
                    num_params = sum(p.numel() for p in model_dict[key].parameters())
                    print(f"      Parameters: {num_params:,}")
                elif isinstance(model_dict[key], dict):
                    print(f"      Type: state_dict")
                    # محاسبه آمار وزن‌ها
                    weights = [v for v in model_dict[key].values() if isinstance(v, torch.Tensor)]
                    if weights:
                        means = [w.mean().item() for w in weights]
                        stds = [w.std().item() for w in weights]
                        print(f"      Mean weight: {np.mean(means):.6f}")
                        print(f"      Mean std: {np.mean(stds):.6f}")
        else:
            print(f"\n  ⚠️ {name}: Unexpected format")
    
    print(f"\n✅ Checkpoint validation complete!")
    return True

if __name__ == "__main__":
    import sys
    
    # اگر از پوشه models/level2 اجرا می‌کنید
    if Path.cwd().name == "level2":
        checkpoint_path = "checkpoint_7000"
    else:
        checkpoint_path = "models/level2/checkpoint_7000"
    
    test_checkpoint(checkpoint_path)
