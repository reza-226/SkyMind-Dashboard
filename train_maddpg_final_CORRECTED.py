# train_maddpg_final_FIXED_V3.py
"""
اسکریپت آموزش MADDPG - نسخه نهایی با تبدیل Tensor
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time

from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent

# =============================================================================
# 1. تنظیمات
# =============================================================================
CONFIG = {
    # محیط
    'n_agents': 3,
    'n_users': 5,
    'dt': 1.0,
    'area_size': 1000.0,
    
    # آموزش
    'n_episodes': 1000,
    'max_steps': 200,
    'batch_size': 128,
    'buffer_size': 100000,
    
    # یادگیری
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'gamma': 0.99,
    'tau': 0.01,
    
    # اکتشاف
    'noise_scale_start': 1.0,
    'noise_scale_end': 0.1,
    'noise_decay': 0.995,
    
    # ذخیره
    'save_interval': 50,
    'log_interval': 10,
    
    # Early stopping
    'patience': 100,
    'min_improvement': 0.01,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# =============================================================================
# 2. توابع کمکی
# =============================================================================

def flatten_state(state_dict):
    """تبدیل dictionary state به vector 38 بعدی"""
    parts = []
    
    # UAV positions (3 UAVs × 2) = 6
    if 'uav_positions' in state_dict:
        parts.append(state_dict['uav_positions'].flatten())
    
    # User positions (5 users × 2) = 10
    if 'user_positions' in state_dict:
        parts.append(state_dict['user_positions'].flatten())
    
    # UAV velocities (3) = 3
    if 'uav_velocities' in state_dict:
        parts.append(state_dict['uav_velocities'])
    
    # UAV headings (3) = 3
    if 'uav_headings' in state_dict:
        parts.append(state_dict['uav_headings'])
    
    # UAV energies (3) = 3
    if 'uav_energies' in state_dict:
        parts.append(state_dict['uav_energies'])
    
    # User data sizes (5) = 5
    if 'user_data_sizes' in state_dict:
        parts.append(state_dict['user_data_sizes'])
    
    # User deadlines (5) = 5
    if 'user_deadlines' in state_dict:
        parts.append(state_dict['user_deadlines'])
    
    # Time remaining (1) = 1
    if 'time_remaining' in state_dict:
        parts.append(np.array([state_dict['time_remaining']]))
    
    # Distances (2) = 2
    if 'distances' in state_dict:
        dist = state_dict['distances']
        if isinstance(dist, np.ndarray):
            parts.append(dist.flatten()[:2])
        else:
            parts.append(np.array([dist, dist]))
    
    # ترکیب همه
    result = np.concatenate(parts)
    
    # اطمینان از 38 بعد
    if len(result) < 38:
        result = np.pad(result, (0, 38 - len(result)), mode='constant')
    elif len(result) > 38:
        result = result[:38]
    
    return result


class ReplayBufferWrapper:
    """
    Wrapper برای Replay Buffer که با Agent سازگار است
    و NumPy arrays را به Tensor تبدیل می‌کند
    """
    
    def __init__(self, max_size, batch_size=128, device='cpu'):
        self.max_size = max_size
        self.batch_size = batch_size
        self.device = device  # برای تبدیل به Tensor
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size=None):
        """
        Sample که با هر دو API سازگار است و NumPy را به Tensor تبدیل می‌کند
        """
        bs = batch_size if batch_size is not None else self.batch_size
        bs = min(bs, len(self.buffer))
        
        if bs == 0:
            return None, None, None, None, None
        
        indices = np.random.choice(len(self.buffer), bs, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        # تبدیل به Tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# 3. آموزش
# =============================================================================

def train():
    print("="*70)
    print("🚀 شروع آموزش MADDPG")
    print("="*70)
    
    # تشخیص device
    device = torch.device(CONFIG['device'])
    print(f"\n🖥️  Device: {device}")
    
    # ساخت محیط
    env = MultiUAVEnv(
        n_agents=CONFIG['n_agents'],
        n_users=CONFIG['n_users'],
        dt=CONFIG['dt'],
        area_size=CONFIG['area_size']
    )
    
    # تعیین ابعاد
    state_dict = env.reset()
    state_sample = flatten_state(state_dict)
    state_dim = len(state_sample)
    action_dim = 4
    
    print(f"\n📊 تنظیمات:")
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Number of agents: {CONFIG['n_agents']}")
    print(f"   Number of users: {CONFIG['n_users']}")
    
    # ساخت Agents
    agents = []
    for i in range(CONFIG['n_agents']):
        agent = MADDPG_Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=CONFIG['n_agents'],
            lr=CONFIG['lr_actor'],
            gamma=CONFIG['gamma']
        )
        # انتقال به device
        if hasattr(agent, 'actor'):
            agent.actor = agent.actor.to(device)
        if hasattr(agent, 'target_actor'):
            agent.target_actor = agent.target_actor.to(device)
        if hasattr(agent, 'critic'):
            agent.critic = agent.critic.to(device)
        if hasattr(agent, 'target_critic'):
            agent.target_critic = agent.target_critic.to(device)
        
        agents.append(agent)
    
    # Replay buffer با device
    replay_buffer = ReplayBufferWrapper(
        max_size=CONFIG['buffer_size'],
        batch_size=CONFIG['batch_size'],
        device=device
    )
    
    # متغیرهای tracking
    episode_rewards = []
    moving_avg_rewards = []
    best_reward = -float('inf')
    patience_counter = 0
    noise_scale = CONFIG['noise_scale_start']
    
    # مسیرهای ذخیره
    model_dir = Path('models')
    results_dir = Path('results')
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    print(f"\n🎯 شروع {CONFIG['n_episodes']} اپیزود...")
    start_time = time.time()
    
    # =============================================================================
    # حلقه اصلی آموزش
    # =============================================================================
    for episode in range(CONFIG['n_episodes']):
        state_dict = env.reset()
        state = flatten_state(state_dict)
        episode_reward = 0
        
        for step in range(CONFIG['max_steps']):
            # انتخاب اکشن (نیازی به تبدیل به Tensor نیست، agent.act خودش انجام می‌دهد)
            actions = []
            for agent in agents:
                action = agent.act(state, noise_scale=noise_scale)
                actions.append(action)
            
            actions = np.array(actions)
            
            # اجرا در محیط
            try:
                next_state_dict, rewards, done, info = env.step(actions)
                next_state = flatten_state(next_state_dict)
            except Exception as e:
                print(f"\n⚠️ خطا در env.step: {e}")
                print(f"   Actions shape: {actions.shape}")
                break
            
            # ذخیره در buffer (به صورت NumPy)
            replay_buffer.push(state, actions, rewards, next_state, done)
            
            # آموزش
            if len(replay_buffer) > CONFIG['batch_size']:
                for i, agent in enumerate(agents):
                    other_agents = [a for j, a in enumerate(agents) if j != i]
                    
                    # حالا sample() خروجی Tensor می‌دهد
                    agent.update(replay_buffer, other_agents)
            
            state = next_state
            episode_reward += np.mean(rewards) if isinstance(rewards, np.ndarray) else rewards
            
            if done:
                break
        
        # ذخیره نتایج اپیزود
        episode_rewards.append(episode_reward)
        
        # محاسبه moving average
        window = min(50, len(episode_rewards))
        moving_avg = np.mean(episode_rewards[-window:])
        moving_avg_rewards.append(moving_avg)
        
        # کاهش noise
        noise_scale = max(CONFIG['noise_scale_end'], 
                         noise_scale * CONFIG['noise_decay'])
        
        # لاگ‌گیری
        if (episode + 1) % CONFIG['log_interval'] == 0:
            elapsed = time.time() - start_time
            print(f"\n📈 Episode {episode + 1}/{CONFIG['n_episodes']}")
            print(f"   Reward: {episode_reward:.2f}")
            print(f"   Moving Avg: {moving_avg:.2f}")
            print(f"   Noise: {noise_scale:.3f}")
            print(f"   Buffer: {len(replay_buffer)}")
            print(f"   Time: {elapsed:.1f}s")
        
        # ذخیره بهترین مدل
        if moving_avg > best_reward + CONFIG['min_improvement']:
            best_reward = moving_avg
            patience_counter = 0
            
            if (episode + 1) % CONFIG['log_interval'] == 0:
                print(f"   💾 ذخیره بهترین مدل (reward: {best_reward:.2f})")
            
            for i, agent in enumerate(agents):
                if hasattr(agent, 'actor'):
                    torch.save(agent.actor.state_dict(), 
                             model_dir / f'best_actor_agent{i}.pt')
                if hasattr(agent, 'critic'):
                    torch.save(agent.critic.state_dict(), 
                             model_dir / f'best_critic_agent{i}.pt')
        else:
            patience_counter += 1
        
        # ذخیره checkpoint
        if (episode + 1) % CONFIG['save_interval'] == 0:
            print(f"   💾 ذخیره checkpoint")
            for i, agent in enumerate(agents):
                if hasattr(agent, 'actor'):
                    torch.save(agent.actor.state_dict(), 
                             model_dir / f'checkpoint_actor_agent{i}_ep{episode+1}.pt')
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\n⚠️ Early stopping (no improvement for {CONFIG['patience']} episodes)")
            break
    
    # =============================================================================
    # پایان آموزش
    # =============================================================================
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("✅ آموزش تکمیل شد!")
    print(f"   زمان کل: {total_time:.1f}s ({total_time/60:.1f} دقیقه)")
    print(f"   بهترین reward: {best_reward:.2f}")
    print("="*70)
    
    # ذخیره metrics
    np.savez(results_dir / 'training_metrics.npz',
             episode_rewards=episode_rewards,
             moving_avg=moving_avg_rewards)
    
    # رسم نمودارها
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    plt.plot(moving_avg_rewards, label='Moving Average (50)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(moving_avg_rewards, linewidth=2, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('Smoothed Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 نمودارها ذخیره شد: {results_dir / 'training_curves.png'}")
    
    return agents, episode_rewards, moving_avg_rewards


# =============================================================================
# اجرا
# =============================================================================
if __name__ == "__main__":
    try:
        agents, rewards, moving_avg = train()
    except Exception as e:
        print(f"\n❌ خطای کلی: {e}")
        import traceback
        traceback.print_exc()
