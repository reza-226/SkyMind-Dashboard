"""
train_multi.py (Compatible Version)
=====================================
سازگار با train_maddpg_complete.py و Dashboard
"""

import argparse
import numpy as np
import torch
import pandas as pd
from collections import deque
from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent
from tqdm import tqdm
import os


def train_maddpg(n_episodes=2000, resume=False, checkpoint_path='models/'):
    """آموزش MADDPG با حفظ سازگاری"""
    
    print("[SkyMind-TPSG] Training Multi-Agent DRL Simulation started...\n")
    
    # محیط
    n_agents = 3
    n_users = 5
    env = MultiUAVEnv(n_agents=n_agents, n_users=n_users)
    
    state_dim = 6
    action_dim = 4
    
    # Agents
    agents = []
    for i in range(n_agents):
        agent = MADDPG_Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            agent_id=i,
            lr=1e-4
        )
        agents.append(agent)
    
    # Replay Buffer
    replay_buffer = deque(maxlen=100000)
    print("[SkyMind-TPSG] ✓ Replay Buffer created")
    
    # Resume logic
    start_episode = 0
    if resume:
        print(f"\n[SkyMind-TPSG] 🔄 Resume mode activated")
        
        # ✅ سازگاری با train_maddpg_complete.py
        legacy_checkpoint = os.path.join(checkpoint_path, 'maddpg_sky_env_1.pth')
        
        if os.path.exists(legacy_checkpoint):
            print("  📂 Legacy checkpoint detected, loading...")
            checkpoint = torch.load(legacy_checkpoint)
            
            for i, agent in enumerate(agents):
                if f'actor_agent{i}' in checkpoint:
                    agent.actor.load_state_dict(checkpoint[f'actor_agent{i}'])
                    agent.critic.load_state_dict(checkpoint[f'critic_agent{i}'])
                    print(f"  ✅ Agent {i} loaded from legacy checkpoint")
        else:
            # فرمت جدید
            for i, agent in enumerate(agents):
                actor_path = os.path.join(checkpoint_path, f'actor_agent{i}.pt')
                critic_path = os.path.join(checkpoint_path, f'critic_agent{i}.pt')
                
                if os.path.exists(actor_path):
                    agent.actor.load_state_dict(torch.load(actor_path))
                    agent.critic.load_state_dict(torch.load(critic_path))
                    print(f"  ✅ Agent {i} loaded")
        
        # بارگذاری شماره episode
        episode_file = os.path.join(checkpoint_path, 'episode.txt')
        if os.path.exists(episode_file):
            with open(episode_file, 'r') as f:
                start_episode = int(f.read().strip())
    
    # Hyperparameters
    batch_size = 128
    epsilon = 0.3 if not resume else 0.1
    epsilon_decay = 0.999
    epsilon_min = 0.05
    
    # Metrics tracking
    rewards_history = []
    critic_losses_history = []
    actor_losses_history = []
    energy_history = []  # ✅ برای سازگاری با Dashboard
    delay_history = []   # ✅ برای سازگاری با Dashboard
    
    # ✅ CSV log file (مطابق train_maddpg_complete.py)
    csv_path = 'data/episodes.csv'
    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists(csv_path) or not resume:
        with open(csv_path, 'w') as f:
            f.write("episode,reward,energy,delay,critic_loss,actor_loss\n")
    
    # حلقه آموزش
    for episode in tqdm(range(start_episode, start_episode + n_episodes), 
                        desc="Training"):
        
        state = env.reset()
        episode_reward = 0
        episode_critic_loss = []
        episode_actor_loss = []
        episode_energy = 0
        episode_delay = 0
        
        for step in range(200):
            # انتخاب اکشن
            actions = []
            for i, agent in enumerate(agents):
                if isinstance(state, dict):
                    agent_state = extract_agent_state(state, i)
                else:
                    agent_state = state[i] if state.ndim > 1 else state
                
                if np.random.rand() < epsilon:
                    action = np.random.uniform(-1, 1, action_dim)
                else:
                    action = agent.select_action(agent_state)
                
                actions.append(action)
            
            actions = np.array(actions)
            
            # اجرای اکشن
            next_state, reward, done, info = env.step(actions)
            
            if isinstance(reward, np.ndarray):
                reward = reward.sum()
            
            episode_reward += reward
            
            # ✅ استخراج energy و delay (اگر موجود باشد)
            if 'energy_total' in info:
                episode_energy += info['energy_total']
            if 'mean_delay' in info:
                episode_delay = info['mean_delay']
            
            # ذخیره در buffer
            replay_buffer.append({
                'state': state,
                'actions': actions,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            # بروزرسانی
            if len(replay_buffer) >= batch_size:
                for agent in agents:
                    critic_loss, actor_loss = agent.update(
                        replay_buffer, 
                        agents, 
                        batch_size=batch_size
                    )
                    
                    if critic_loss is not None:
                        episode_critic_loss.append(critic_loss)
                    if actor_loss is not None:
                        episode_actor_loss.append(actor_loss)
            
            state = next_state
            
            if done:
                break
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # ذخیره متریک‌ها
        rewards_history.append(episode_reward)
        energy_history.append(episode_energy)
        delay_history.append(episode_delay)
        
        avg_critic_loss = np.mean(episode_critic_loss) if episode_critic_loss else 0
        avg_actor_loss = np.mean(episode_actor_loss) if episode_actor_loss else 0
        
        critic_losses_history.append(avg_critic_loss)
        actor_losses_history.append(avg_actor_loss)
        
        # ✅ ذخیره در CSV (مطابق فرمت قبلی)
        with open(csv_path, 'a') as f:
            f.write(f"{episode + 1},{episode_reward:.4f},{episode_energy:.4f},"
                    f"{episode_delay:.4f},{avg_critic_loss:.6f},{avg_actor_loss:.6f}\n")
        
        # Checkpoint
        if (episode + 1) % 100 == 0:
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # ✅ ذخیره در دو فرمت (سازگاری)
            # فرمت 1: فایل‌های جداگانه (جدید)
            for i, agent in enumerate(agents):
                torch.save(
                    agent.actor.state_dict(),
                    os.path.join(checkpoint_path, f'actor_agent{i}.pt')
                )
                torch.save(
                    agent.critic.state_dict(),
                    os.path.join(checkpoint_path, f'critic_agent{i}.pt')
                )
            
            # فرمت 2: فایل واحد (قدیمی - برای Dashboard)
            legacy_dict = {}
            for i, agent in enumerate(agents):
                legacy_dict[f'actor_agent{i}'] = agent.actor.state_dict()
                legacy_dict[f'critic_agent{i}'] = agent.critic.state_dict()
            
            torch.save(legacy_dict, 
                      os.path.join(checkpoint_path, 'maddpg_sky_env_1.pth'))
            
            # ذخیره شماره episode
            with open(os.path.join(checkpoint_path, 'episode.txt'), 'w') as f:
                f.write(str(episode + 1))
            
            print(f"\n💾 Checkpoint saved (Episode {episode + 1})")
            print(f"   Reward: {episode_reward:.4f}")
            print(f"   Energy: {episode_energy:.4f}")
            print(f"   Delay: {episode_delay:.4f}")
    
    # ذخیره نهایی
    print("\n[SkyMind-TPSG] ✅ Training completed")
    
    # ✅ NPZ با فرمت کامل (سازگار با Dashboard)
    np.savez(
        'results/training_metrics.npz',
        rewards=rewards_history,
        critic_losses=critic_losses_history,  # ✅ نام دقیق
        actor_losses=actor_losses_history,
        energy=energy_history,  # ✅ برای Dashboard
        delay=delay_history      # ✅ برای Dashboard
    )
    print("💾 Metrics saved: results/training_metrics.npz")
    
    return agents, rewards_history


def extract_agent_state(state_dict, agent_idx, state_dim=6):
    """استخراج state عامل"""
    if isinstance(state_dict, dict):
        agent_state = np.concatenate([
            state_dict['uav_positions'][agent_idx],
            [state_dict['uav_velocities'][agent_idx]],
            [state_dict['uav_angles'][agent_idx]],
            [state_dict['energy'][agent_idx]],
            [state_dict['distances'][agent_idx]]
        ])
        return agent_state
    return state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='models/')
    
    args = parser.parse_args()
    
    train_maddpg(
        n_episodes=args.episodes,
        resume=args.resume,
        checkpoint_path=args.checkpoint
    )
