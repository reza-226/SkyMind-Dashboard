"""
اسکریپت نهایی آموزش MADDPG با پشتیبانی کامل از:
- محیط‌های Dictionary-based (agent_0, agent_1, ...)
- ساختار فایل واقعی پروژه (agents/maddpg_wrapper.py)
- پشتیبانی 3D CollisionChecker
- مدیریت خودکار Discrete/Continuous action spaces
- ذخیره و بارگذاری مدل‌ها
- لاگ‌گیری TensorBoard
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

# 🔧 Patch CollisionChecker برای سازگاری 3D
sys.path.insert(0, str(Path(__file__).parent / "core"))
from collision_checker_patch import patch_collision_checker
patch_collision_checker()  # اعمال پچ قبل از ساخت محیط

# اضافه کردن مسیر پروژه
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'core'))
sys.path.insert(0, str(project_root / 'agents'))

# Import محیط
from core.env_multi import MultiUAVEnv

# Import Agent
from agents.maddpg_wrapper import MADDPGAgent

# Import یوتیلیتی‌ها
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️ TensorBoard not available. Install with: pip install tensorboard")


class ReplayBuffer:
    """
    بافر تجربه برای ذخیره و نمونه‌برداری تجربیات
    """
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self, 
        states: Dict[str, np.ndarray], 
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_states: Dict[str, np.ndarray],
        dones: Dict[str, bool]
    ):
        """افزودن یک تجربه"""
        self.buffer.append((states, actions, rewards, next_states, dones))
    
    def sample(self, batch_size: int) -> Tuple:
        """نمونه‌برداری تصادفی"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for idx in indices:
            states, actions, rewards, next_states, dones = self.buffer[idx]
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_next_states.append(next_states)
            batch_dones.append(dones)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
    def __len__(self):
        return len(self.buffer)


def create_env(config: Optional[Dict] = None) -> MultiUAVEnv:
    """
    ساخت محیط با پارامترهای صحیح
    """
    default_config = {
        'num_uavs': 3,           # پارامتر صحیح (نه n_uavs)
        'map_size': 100,
        'num_obstacles': 10,
        'max_steps': 500,
        'render_mode': None
    }
    
    if config:
        default_config.update(config)
    
    print(f"🏗️ Creating environment with config: {default_config}")
    
    try:
        env = MultiUAVEnv(**default_config)
        
        # افزودن alias برای سازگاری
        if hasattr(env, 'num_uavs') and not hasattr(env, 'n_agents'):
            env.n_agents = env.num_uavs
        
        print(f"✅ Environment created: {env.num_uavs} UAVs")
        return env
        
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


def get_state_action_dims(env: MultiUAVEnv) -> Tuple[int, int]:
    """
    استخراج ابعاد state و action از محیط
    """
    # دریافت یک نمونه state
    states, _ = env.reset()
    
    # استخراج state_dim از اولین agent
    first_agent_key = list(states.keys())[0]
    sample_state = states[first_agent_key]
    state_dim = sample_state.flatten().shape[0]
    
    # استخراج action_dim
    sample_action_space = env.action_space
    
    if hasattr(sample_action_space, 'n'):
        # Discrete action space
        act_dim = sample_action_space.n
        print(f"📊 Discrete action space detected: {act_dim} actions")
    elif hasattr(sample_action_space, 'shape'):
        # Continuous action space
        act_dim = sample_action_space.shape[0] if len(sample_action_space.shape) > 0 else 2
        print(f"📊 Continuous action space detected: {act_dim}D")
    else:
        # Fallback: فرض کن 2D continuous
        act_dim = 2
        print(f"⚠️ Unknown action space, assuming 2D continuous")
    
    print(f"📐 State dim: {state_dim}, Action dim: {act_dim}")
    
    return state_dim, act_dim


def train_maddpg(
    env: MultiUAVEnv,
    n_episodes: int = 1000,
    max_steps: int = 500,
    batch_size: int = 64,
    buffer_capacity: int = 100000,
    update_freq: int = 100,
    save_freq: int = 100,
    log_dir: str = "runs",
    model_dir: str = "models"
):
    """
    حلقه اصلی آموزش MADDPG
    """
    
    # ایجاد دایرکتوری‌ها
    Path(log_dir).mkdir(exist_ok=True)
    Path(model_dir).mkdir(exist_ok=True)
    
    # TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(f"{log_dir}/maddpg_{timestamp}")
    
    # استخراج ابعاد
    state_dim, act_dim = get_state_action_dims(env)
    n_agents = env.num_uavs
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting MADDPG Training")
    print(f"{'='*70}")
    print(f"Agents: {n_agents}")
    print(f"State dim: {state_dim}, Action dim: {act_dim}")
    print(f"Episodes: {n_episodes}, Max steps: {max_steps}")
    print(f"Batch size: {batch_size}, Buffer: {buffer_capacity}")
    print(f"{'='*70}\n")
    
    # ساخت Agents
    agents = {}
    for i in range(n_agents):
        agent_id = f"agent_{i}"
        agents[agent_id] = MADDPGAgent(
            state_dim=state_dim,
            action_dim=act_dim,
            n_agents=n_agents,
            agent_id=i,
            hidden_dim=256,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.99,
            tau=0.01
        )
    
    # Replay Buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    
    # آمار آموزش
    episode_rewards = []
    episode_losses = []
    best_avg_reward = -float('inf')
    
    # حلقه اصلی آموزش
    for episode in range(n_episodes):
        states, _ = env.reset()
        episode_reward = {agent_id: 0.0 for agent_id in agents.keys()}
        episode_loss = {agent_id: [] for agent_id in agents.keys()}
        
        for step in range(max_steps):
            # انتخاب actions
            actions = {}
            for agent_id, agent in agents.items():
                state = states[agent_id].flatten()
                
                # اضافه کردن نویز برای اکتشاف
                action = agent.select_action(state, add_noise=True)
                actions[agent_id] = action
            
            # اجرای action در محیط
            next_states, rewards, dones, truncated, _ = env.step(actions)
            
            # ذخیره در بافر
            replay_buffer.add(states, actions, rewards, next_states, dones)
            
            # به‌روزرسانی rewards
            for agent_id in agents.keys():
                episode_reward[agent_id] += rewards[agent_id]
            
            # آموزش agents (اگر بافر کافی باشد)
            if len(replay_buffer) > batch_size:
                # نمونه‌برداری از بافر
                batch = replay_buffer.sample(batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch
                
                # آموزش هر agent
                for agent_id, agent in agents.items():
                    # استخراج داده‌های این agent
                    agent_states = np.array([s[agent_id].flatten() for s in batch_states])
                    agent_actions = np.array([a[agent_id] for a in batch_actions])
                    agent_rewards = np.array([r[agent_id] for r in batch_rewards])
                    agent_next_states = np.array([s[agent_id].flatten() for s in batch_next_states])
                    agent_dones = np.array([d[agent_id] for d in batch_dones])
                    
                    # جمع‌آوری اطلاعات سایر agents
                    all_states = np.array([[s[aid].flatten() for aid in agents.keys()] 
                                          for s in batch_states])
                    all_actions = np.array([[a[aid] for aid in agents.keys()] 
                                           for a in batch_actions])
                    all_next_states = np.array([[s[aid].flatten() for aid in agents.keys()] 
                                                for s in batch_next_states])
                    
                    # آموزش
                    critic_loss, actor_loss = agent.update(
                        agent_states,
                        agent_actions,
                        agent_rewards,
                        agent_next_states,
                        agent_dones,
                        all_states,
                        all_actions,
                        all_next_states
                    )
                    
                    episode_loss[agent_id].append((critic_loss, actor_loss))
            
            # به‌روزرسانی state
            states = next_states
            
            # بررسی پایان
            if all(dones.values()) or all(truncated.values()):
                break
        
        # آمار episode
        avg_reward = np.mean(list(episode_reward.values()))
        episode_rewards.append(avg_reward)
        
        # محاسبه میانگین loss
        avg_losses = {}
        for agent_id in agents.keys():
            if episode_loss[agent_id]:
                critic_losses, actor_losses = zip(*episode_loss[agent_id])
                avg_losses[agent_id] = {
                    'critic': np.mean(critic_losses),
                    'actor': np.mean(actor_losses)
                }
        
        # لاگ‌گیری
        if writer:
            writer.add_scalar('Train/AverageReward', avg_reward, episode)
            for agent_id, losses in avg_losses.items():
                writer.add_scalar(f'Train/{agent_id}/CriticLoss', losses['critic'], episode)
                writer.add_scalar(f'Train/{agent_id}/ActorLoss', losses['actor'], episode)
        
        # چاپ پیشرفت
        if (episode + 1) % 10 == 0:
            recent_avg = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Recent 100: {recent_avg:.2f} | "
                  f"Buffer: {len(replay_buffer)}")
        
        # ذخیره بهترین مدل
        if (episode + 1) % save_freq == 0:
            recent_avg = np.mean(episode_rewards[-100:])
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                for agent_id, agent in agents.items():
                    agent.save(f"{model_dir}/{agent_id}_best.pth")
                print(f"💾 Best model saved! Avg reward: {best_avg_reward:.2f}")
        
        # ذخیره چک‌پوینت
        if (episode + 1) % save_freq == 0:
            for agent_id, agent in agents.items():
                agent.save(f"{model_dir}/{agent_id}_ep{episode+1}.pth")
    
    # بستن writer
    if writer:
        writer.close()
    
    # رسم نمودار rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
    
    # میانگین متحرک
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window)/window, 
                                mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), 
                moving_avg, 
                'r-', 
                linewidth=2, 
                label=f'{window}-Episode Moving Average')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('MADDPG Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{log_dir}/training_rewards.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"✅ Training completed!")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Final average reward (last 100 eps): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"{'='*70}\n")
    
    return agents, episode_rewards


def main():
    """
    تابع اصلی
    """
    # تنظیمات محیط
    env_config = {
        'num_uavs': 3,
        'map_size': 100,
        'num_obstacles': 10,
        'max_steps': 500,
        'render_mode': None
    }
    
    # ساخت محیط
    env = create_env(env_config)
    
    # تنظیمات آموزش
    train_config = {
        'n_episodes': 1000,
        'max_steps': 500,
        'batch_size': 64,
        'buffer_capacity': 100000,
        'update_freq': 100,
        'save_freq': 100,
        'log_dir': 'runs',
        'model_dir': 'models'
    }
    
    # شروع آموزش
    try:
        agents, rewards = train_maddpg(env, **train_config)
        print("🎉 Training finished successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()
        print("🔒 Environment closed")


if __name__ == "__main__":
    main()
