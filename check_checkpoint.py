import os
import glob
import torch

BASE_DIR = r"\content\drive\MyDrive\uav_mec\results"

for level in range(1, 5):
    level_dir = os.path.join(BASE_DIR, f"level{level}")
    
    if not os.path.exists(level_dir):
        print(f"❌ Level {level}: Directory not found")
        continue
    
    # Find latest run
    runs = [d for d in os.listdir(level_dir) if os.path.isdir(os.path.join(level_dir, d))]
    if not runs:
        print(f"❌ Level {level}: No runs found")
        continue
    
    latest_run = sorted(runs)[-1]
    checkpoint_dir = os.path.join(level_dir, latest_run, "checkpoints")
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Level {level}: No checkpoints")
        continue
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_episode_*.pt"))
    
    if not checkpoints:
        print(f"❌ Level {level}: No checkpoint files")
        continue
    
    # آخرین checkpoint
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
    latest_checkpoint = checkpoints[-1]
    episode_num = int(latest_checkpoint.split('_')[-1].replace('.pt', ''))
    
    # بارگذاری و بررسی
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    print(f"✅ Level {level}:")
    print(f"   Run: {latest_run}")
    print(f"   Episode: {episode_num}")
    print(f"   Agents: {list(checkpoint['agents'].keys())}")
    print(f"   Path: {latest_checkpoint}")
    print()
