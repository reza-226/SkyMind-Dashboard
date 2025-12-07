#!/usr/bin/env python3
"""
اسکریپت آموزش MADDPG برای UAV-MEC Task Offloading
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import json
import argparse

# ==================== تنظیم مسیر پروژه ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"🐍 Python Path: {sys.path[:3]}")

# ==================== Import ماژول‌ها ====================
try:
    from core.agent.maddpg_agent import MADDPGAgent
    from core.env.environment import UAVMECEnvironment
    from models.dag import DAG  # ✅ استفاده از کلاس DAG موجود
    print("✅ All modules imported successfully!\n")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n🔍 فایل‌های زیر را بررسی کنید:")
    print("   1. core/agent/maddpg_agent.py")
    print("   2. core/env/environment.py")
    print("   3. models/dag.py")
    sys.exit(1)


def train_maddpg(args):
    """تابع اصلی آموزش MADDPG"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Create directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Models will be saved to: {checkpoint_dir.absolute()}\n")
    
    print("=" * 60)
    print("🚀 Starting MADDPG Training")
    print("=" * 60)
    
    try:
        # ==================== ایجاد محیط و Agent ====================
        print("\n📦 Initializing Environment and Agent...")
        
        # ✅ فقط device و max_steps
        env = UAVMECEnvironment(
            device=str(device),
            max_steps=args.max_steps
        )
        
        # ایجاد Agent
        agent = MADDPGAgent(
            state_dim=537,
            offload_dim=5,
            continuous_dim=6,
            hidden=512,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            tau=args.tau,
            buffer_size=args.buffer_size,
            device=device
        )
        
        print("✅ Initialization complete!\n")
        
        # ==================== متغیرهای آموزش ====================
        total_rewards = []
        best_reward = float('-inf')
        noise_std = args.noise_std
        
        # ==================== حلقه آموزش ====================
        print(f"🎯 Training for {args.episodes} episodes...\n")
        
        for episode in range(args.episodes):
            # Generate DAG using the DAG class
            dag = DAG.generate_random_dag(
                num_nodes=10, 
                edge_prob=0.3,
                device=str(device)
            )
            
            state = env.reset(dag)
            
            episode_reward = 0.0
            done = False
            steps = 0
            
            # Progress bar
            pbar = tqdm(
                range(args.max_steps),
                desc=f"Ep {episode+1}/{args.episodes}",
                leave=False
            )
            
            for step in pbar:
                # Select action
                action = agent.select_action(state, noise_std=noise_std, training=True)
                
                # Environment step
                next_state, reward, done, _ = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Update agent
                if len(agent.buffer) >= args.min_buffer_size and steps % args.update_freq == 0:
                    for _ in range(args.updates_per_step):
                        actor_loss, critic_loss = agent.update(batch_size=args.batch_size)
                        
                        if actor_loss is not None:
                            pbar.set_postfix({
                                'reward': f'{reward:.2f}',
                                'a_loss': f'{actor_loss:.4f}',
                                'c_loss': f'{critic_loss:.4f}'
                            })
                
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            
            # Decay noise
            noise_std = max(args.min_noise, noise_std * args.noise_decay)
            
            # Logging
            if episode % args.log_interval == 0:
                avg_reward = np.mean(total_rewards[-args.log_interval:]) if len(total_rewards) >= args.log_interval else np.mean(total_rewards)
                print(f"\n📊 Episode {episode:4d} | Steps: {steps:3d} | "
                      f"Reward: {episode_reward:8.2f} | "
                      f"Avg: {avg_reward:8.2f} | "
                      f"Noise: {noise_std:.4f}")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                save_path = checkpoint_dir / f"best_ep{episode}.pt"
                agent.save(save_path)
                print(f"🏆 New best model saved! Reward: {best_reward:.2f}")
            
            # Periodic checkpoint
            if (episode + 1) % args.save_interval == 0:
                save_path = checkpoint_dir / f"checkpoint_ep{episode+1}.pt"
                agent.save(save_path)
                print(f"💾 Checkpoint saved at episode {episode+1}")
        
        # ==================== ذخیره مدل نهایی ====================
        final_path = checkpoint_dir / "final_model.pt"
        agent.save(final_path)
        
        # ذخیره نتایج
        results = {
            "total_episodes": args.episodes,
            "total_rewards": total_rewards,
            "best_reward": best_reward,
            "final_avg_100": float(np.mean(total_rewards[-100:])) if len(total_rewards) >= 100 else float(np.mean(total_rewards)),
            "config": vars(args)
        }
        
        results_path = output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)
        print(f"📊 Best Reward: {best_reward:.2f}")
        print(f"📊 Final Avg (last 100): {np.mean(total_rewards[-100:]):.2f}")
        print(f"📈 Results saved to: {results_path}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MADDPG Training for UAV-MEC")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    
    # Agent hyperparameters
    parser.add_argument("--lr_actor", type=float, default=0.0001, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=0.001, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter")
    
    # Exploration
    parser.add_argument("--noise_std", type=float, default=0.5, help="Initial noise std")
    parser.add_argument("--noise_decay", type=float, default=0.995, help="Noise decay rate")
    parser.add_argument("--min_noise", type=float, default=0.01, help="Minimum noise")
    
    # Training settings
    parser.add_argument("--update_freq", type=int, default=5, help="Update frequency")
    parser.add_argument("--updates_per_step", type=int, default=5, help="Updates per step")
    parser.add_argument("--min_buffer_size", type=int, default=1000, help="Min buffer before training")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--save_interval", type=int, default=50, help="Save interval")
    parser.add_argument("--output_dir", type=str, default="output/training_runs", help="Output directory")
    
    args = parser.parse_args()
    train_maddpg(args)
