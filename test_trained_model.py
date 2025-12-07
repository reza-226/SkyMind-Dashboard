# test_trained_model.py (نسخه نهایی - Robust برای همه فرمت‌های config)
"""
تست مدل آموزش‌دیده با MADDPG
✅ سازگار کامل با train_4layer_3level.py
✅ مقاوم در برابر فرمت‌های مختلف config
"""
import torch
import numpy as np
import sys
import os
from pathlib import Path
import json

# اضافه کردن مسیرها
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from environments.uav_mec_env import UAVMECEnvironment
from models.actor_critic.maddpg_agent import MADDPGAgent

def test_model(model_path, num_episodes=10, render=False):
    """
    تست مدل آموزش‌دیده
    
    Args:
        model_path: مسیر فایل مدل
        num_episodes: تعداد اپیزودهای تست
        render: نمایش محیط (اگر ممکن باشد)
    """
    print(f"\n{'='*70}")
    print(f"🧪 Testing Trained MADDPG Model")
    print(f"{'='*70}")
    print(f"📁 Model: {model_path}")
    print(f"🎲 Episodes: {num_episodes}")
    print(f"{'='*70}\n")
    
    # بارگذاری config از پوشه مدل
    model_dir = Path(model_path).parent
    config_path = model_dir / 'config.json'
    
    env_config = None
    state_dim = None
    action_dim = None
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded config from: {config_path}")
            print(f"📋 Config keys: {list(config.keys())}\n")
            
            # استخراج env_config (فرمت‌های مختلف)
            if 'env_config' in config:
                env_config = config['env_config']
            elif 'config' in config:
                env_config = config['config']
            
            # استخراج dimensions (فرمت‌های مختلف)
            if 'dimensions' in config:
                state_dim = config['dimensions'].get('state_dim')
                action_dim = config['dimensions'].get('action_dim')
            elif 'state_dim' in config:
                state_dim = config['state_dim']
                action_dim = config.get('action_dim', 7)
            
            if state_dim:
                print(f"📐 Config Dimensions:")
                print(f"   State:  {state_dim}")
                print(f"   Action: {action_dim}\n")
            
        except Exception as e:
            print(f"⚠️  Could not parse config: {e}")
            config_path = None
    
    # اگر config نبود یا ناقص بود، از defaults استفاده کن
    if env_config is None:
        print(f"⚠️  Using default environment config")
        env_config = {
            'num_uavs': 3,
            'num_devices': 5,
            'num_edge_servers': 2,
            'grid_size': 500.0,
            'max_steps': 50,
        }
    
    if action_dim is None:
        action_dim = 7
    
    # ساخت محیط
    print(f"🌍 Creating environment with config:")
    for k, v in env_config.items():
        print(f"   {k}: {v}")
    
    env = UAVMECEnvironment(**env_config)
    
    # تشخیص ابعاد از محیط
    print(f"\n🔍 Detecting environment dimensions...")
    dummy_state = env.reset()
    if isinstance(dummy_state, tuple):
        dummy_state = dummy_state[0]
    
    detected_state_dim = len(dummy_state) if isinstance(dummy_state, np.ndarray) else dummy_state.shape[0]
    
    # استفاده از ابعاد config یا detected
    if state_dim is None:
        state_dim = detected_state_dim
        print(f"   Using detected state_dim: {state_dim}")
    else:
        if state_dim != detected_state_dim:
            print(f"   ⚠️  WARNING: Config state_dim ({state_dim}) != detected ({detected_state_dim})")
            print(f"   ❓ Which one to use?")
            # استفاده از config برای سازگاری با مدل آموزش‌دیده
            print(f"   → Using CONFIG value: {state_dim} (model was trained with this)")
        else:
            print(f"   ✅ Dimensions match: {state_dim}")
    
    print(f"\n{'='*60}")
    print(f"📐 Final Dimensions:")
    print(f"   State:  {state_dim}")
    print(f"   Action: {action_dim}")
    print(f"{'='*60}\n")
    
    # ساخت Agent (با همان ساختار train_4layer_3level.py)
    print(f"🤖 Creating MADDPGAgent...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    try:
        agent = MADDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=512,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.99,
            tau=0.01
        )
        print(f"   ✅ Agent created successfully")
    except Exception as e:
        print(f"   ❌ Error creating agent: {e}")
        return None
    
    # بارگذاری وزن‌ها
    print(f"\n📦 Loading model weights...")
    try:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # بررسی ساختار checkpoint
        print(f"   Keys in checkpoint: {list(checkpoint.keys())}")
        
        # Load state_dicts
        loaded = False
        
        # روش 1: actor_state_dict & critic_state_dict
        if 'actor_state_dict' in checkpoint:
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            print(f"   ✅ Loaded actor_state_dict")
            if 'critic_state_dict' in checkpoint:
                agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                print(f"   ✅ Loaded critic_state_dict")
            loaded = True
        
        # روش 2: actor & critic
        elif 'actor' in checkpoint and 'critic' in checkpoint:
            agent.actor.load_state_dict(checkpoint['actor'])
            agent.critic.load_state_dict(checkpoint['critic'])
            print(f"   ✅ Loaded actor & critic")
            loaded = True
        
        # روش 3: model_state_dict
        elif 'model_state_dict' in checkpoint:
            agent.actor.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Loaded model_state_dict")
            loaded = True
        
        # روش 4: مستقیم (فقط actor)
        else:
            try:
                agent.actor.load_state_dict(checkpoint)
                print(f"   ✅ Loaded actor (direct)")
                loaded = True
            except:
                pass
        
        if not loaded:
            raise ValueError("Could not load model weights - unknown checkpoint format")
        
        print(f"✅ Model loaded successfully!\n")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"\n🔍 Checkpoint structure:")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                for k, v in list(checkpoint.items())[:10]:
                    if isinstance(v, dict):
                        print(f"   {k}: dict with {len(v)} keys")
                    elif hasattr(v, 'shape'):
                        print(f"   {k}: tensor {v.shape}")
                    else:
                        print(f"   {k}: {type(v)}")
        except:
            pass
        return None
    
    # Set to eval mode
    agent.actor.eval()
    
    # تست episodes
    print(f"{'='*70}")
    print(f"🎮 Running Test Episodes")
    print(f"{'='*70}\n")
    
    test_rewards = []
    test_lengths = []
    
    for ep in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        # چک ابعاد state
        current_state_dim = len(state)
        if current_state_dim != state_dim:
            print(f"⚠️  Episode {ep+1}: State dimension mismatch!")
            print(f"   Expected: {state_dim}, Got: {current_state_dim}")
            print(f"   Attempting to adapt...")
            
            # سعی در تطبیق ابعاد
            if current_state_dim > state_dim:
                state = state[:state_dim]
                print(f"   → Truncated to {state_dim}")
            else:
                state = np.pad(state, (0, state_dim - current_state_dim))
                print(f"   → Padded to {state_dim}")
        
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # Select action (greedy, no exploration)
            try:
                with torch.no_grad():
                    action = agent.select_action(state, noise=0.0)
            except Exception as e:
                print(f"   ❌ Error selecting action: {e}")
                break
            
            # Execute in environment
            try:
                result = env.step(action)
            except Exception as e:
                print(f"   ❌ Error in env.step: {e}")
                break
            
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                print(f"   ⚠️  Unexpected step result format: {len(result)} items")
                break
            
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            
            # تطبیق ابعاد next_state
            if len(next_state) != state_dim:
                if len(next_state) > state_dim:
                    next_state = next_state[:state_dim]
                else:
                    next_state = np.pad(next_state, (0, state_dim - len(next_state)))
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            # Safety limit
            if step_count > 1000:
                print(f"   ⚠️  Episode {ep+1} too long (>1000 steps), breaking...")
                break
        
        test_rewards.append(episode_reward)
        test_lengths.append(step_count)
        
        print(f"Episode {ep+1:2d}/{num_episodes} │ Steps: {step_count:3d} │ Reward: {episode_reward:9.2f}")
    
    # آمار نهایی
    print(f"\n{'='*70}")
    print(f"📊 Test Results Summary")
    print(f"{'='*70}")
    print(f"  Episodes:      {num_episodes}")
    print(f"  Mean Reward:   {np.mean(test_rewards):9.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Min Reward:    {np.min(test_rewards):9.2f}")
    print(f"  Max Reward:    {np.max(test_rewards):9.2f}")
    print(f"  Mean Length:   {np.mean(test_lengths):.1f} steps")
    print(f"{'='*70}\n")
    
    return test_rewards, test_lengths


def compare_with_random(env_config, num_episodes=10):
    """
    مقایسه با random policy
    """
    print(f"\n{'='*70}")
    print(f"🎲 Testing Random Policy (Baseline)")
    print(f"{'='*70}\n")
    
    env = UAVMECEnvironment(**env_config)
    
    random_rewards = []
    random_lengths = []
    
    for ep in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            # Random action
            action = np.random.uniform(-1, 1, size=7)
            
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                break
            
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            
            episode_reward += reward
            step_count += 1
            state = next_state
            
            if step_count > 1000:
                break
        
        random_rewards.append(episode_reward)
        random_lengths.append(step_count)
        
        print(f"Episode {ep+1:2d}/{num_episodes} │ Steps: {step_count:3d} │ Reward: {episode_reward:9.2f}")
    
    print(f"\n{'='*70}")
    print(f"📊 Random Policy Results")
    print(f"{'='*70}")
    print(f"  Mean Reward:   {np.mean(random_rewards):9.2f} ± {np.std(random_rewards):.2f}")
    print(f"  Min Reward:    {np.min(random_rewards):9.2f}")
    print(f"  Max Reward:    {np.max(random_rewards):9.2f}")
    print(f"  Mean Length:   {np.mean(random_lengths):.1f} steps")
    print(f"{'='*70}\n")
    
    return random_rewards, random_lengths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained MADDPG model')
    parser.add_argument('--model', type=str,
                       default='results/4layer_3level/level_1/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes')
    parser.add_argument('--compare-random', action='store_true',
                       help='Also run random policy for comparison')
    args = parser.parse_args()
    
    # Test trained model
    results = test_model(args.model, num_episodes=args.episodes)
    
    if results is None:
        print("❌ Model testing failed!")
        exit(1)
    
    trained_rewards, trained_lengths = results
    
    # Compare with random if requested
    if args.compare_random:
        # Load env config
        model_dir = Path(args.model).parent
        config_path = model_dir / 'config.json'
        
        env_config = None
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                env_config = config.get('env_config') or config.get('config')
            except:
                pass
        
        if env_config is None:
            env_config = {
                'num_uavs': 3,
                'num_devices': 5,
                'num_edge_servers': 2,
                'grid_size': 500.0,
                'max_steps': 50,
            }
        
        random_rewards, random_lengths = compare_with_random(env_config, num_episodes=args.episodes)
        
        # Final comparison
        improvement = np.mean(trained_rewards) - np.mean(random_rewards)
        improvement_pct = (improvement / abs(np.mean(random_rewards))) * 100 if np.mean(random_rewards) != 0 else 0
        
        print(f"\n{'='*70}")
        print(f"📈 Performance Comparison")
        print(f"{'='*70}")
        print(f"  Trained Mean:     {np.mean(trained_rewards):9.2f}")
        print(f"  Random Mean:      {np.mean(random_rewards):9.2f}")
        print(f"  Improvement:      {improvement:9.2f} ({improvement_pct:+.1f}%)")
        print(f"{'='*70}")
        
        if improvement > 0:
            print(f"✅ Trained model is BETTER than random policy!")
        elif improvement < -100:
            print(f"❌ Trained model is significantly WORSE than random!")
        else:
            print(f"⚠️  Trained model is similar to random (needs more training)")
        
        print(f"{'='*70}\n")
    
    print("✅ Testing complete!")
