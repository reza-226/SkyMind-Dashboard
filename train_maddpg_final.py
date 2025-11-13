"""
train_maddpg_final.py (نسخه v3 - سازگار با محیط سفارشی)
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent

# پارامترهای آموزش
EPISODES = 5000
MAX_STEPS = 200
BATCH_SIZE = 256
SAVE_INTERVAL = 100
EARLY_STOP_PATIENCE = 200

os.makedirs('results/maddpg_training', exist_ok=True)
os.makedirs('models/checkpoints', exist_ok=True)

def get_env_dimensions(env):
    """استخراج ابعاد از محیط سفارشی"""
    
    # تعداد عامل‌ها
    num_agents = None
    for attr in ['num_uavs', 'n_uavs', 'num_agents', 'n_agents']:
        if hasattr(env, attr):
            num_agents = getattr(env, attr)
            print(f"   ✓ تعداد عامل‌ها ({attr}): {num_agents}")
            break
    
    if num_agents is None:
        num_agents = 3  # پیش‌فرض
        print(f"   ⚠️ تعداد عامل‌ها: {num_agents} (پیش‌فرض)")
    
    # بعد حالت - تست با reset
    state, _ = env.reset()
    
    if isinstance(state, np.ndarray):
        if state.ndim == 1:
            state_dim = len(state)
        elif state.ndim == 2:
            state_dim = state.shape[1]  # برای multi-agent
        else:
            state_dim = np.prod(state.shape)
    else:
        # اگر state یک لیست یا tuple است
        state_dim = len(state) if hasattr(state, '__len__') else 10  # پیش‌فرض
    
    print(f"   ✓ بعد حالت: {state_dim} (از reset استخراج شد)")
    
    # بعد عمل - تست
    action_dim = None
    for attr in ['action_dim', 'action_space_dim']:
        if hasattr(env, attr):
            action_dim = getattr(env, attr)
            print(f"   ✓ بعد عمل ({attr}): {action_dim}")
            break
    
    if action_dim is None:
        action_dim = 3  # پیش‌فرض: [offload_ratio, cpu_freq, tx_power]
        print(f"   ⚠️ بعد عمل: {action_dim} (پیش‌فرض)")
    
    return num_agents, state_dim, action_dim

def train_maddpg():
    """آموزش MADDPG"""
    
    print("="*60)
    print("🚀 شروع آموزش MADDPG")
    print("="*60)
    
    # ایجاد محیط
    print("\n📌 در حال ایجاد محیط...")
    env = MultiUAVEnv()
    print("   ✓ محیط ایجاد شد")
    
    # استخراج ابعاد
    print("\n📌 استخراج ابعاد محیط...")
    num_agents, state_dim, action_dim = get_env_dimensions(env)
    
    # ایجاد Agent
    print("\n📌 در حال ایجاد MADDPG Agent...")
    agent = MADDPG_Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.01,
        buffer_size=1000000
    )
    print("   ✓ Agent آماده است")
    
    # متغیرهای آموزش
    episode_rewards = []
    episode_delays = []
    episode_energies = []
    best_reward = float('-inf')
    no_improvement = 0
    
    # حلقه آموزش
    print("\n" + "="*60)
    print("🎓 شروع حلقه آموزش...")
    print("="*60 + "\n")
    
    pbar = tqdm(range(1, EPISODES + 1), desc="Training")
    
    for episode in pbar:
        state, _ = env.reset()
        
        # تبدیل state به numpy array در صورت نیاز
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # Flatten کردن state در صورت نیاز
        if state.ndim > 1:
            state = state.flatten()
        
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0
        
        for step in range(MAX_STEPS):
            # انتخاب عمل
            actions = agent.select_actions(state, add_noise=True)
            
            # Reshape برای محیط (اگر نیاز باشد)
            actions_for_env = actions.reshape(num_agents, action_dim)
            
            # اجرا در محیط
            next_state, reward, done, truncated, info = env.step(actions_for_env)
            
            # تبدیل next_state
            if not isinstance(next_state, np.ndarray):
                next_state = np.array(next_state)
            if next_state.ndim > 1:
                next_state = next_state.flatten()
            
            # ذخیره در بافر
            agent.store_transition(state, actions, reward, next_state, done)
            
            # آموزش
            if len(agent.memory) > BATCH_SIZE:
                agent.update(BATCH_SIZE)
            
            # آپدیت
            episode_reward += reward
            episode_delay += info.get('delay_total', info.get('delay', 0))
            episode_energy += info.get('energy_total', info.get('energy', 0))
            
            state = next_state
            
            if done or truncated:
                break
        
        # ذخیره متریک‌ها
        episode_rewards.append(episode_reward)
        episode_delays.append(episode_delay)
        episode_energies.append(episode_energy)
        
        # آپدیت progress bar
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        pbar.set_postfix({
            'Reward': f'{episode_reward:.2e}',
            'Avg100': f'{avg_reward:.2e}',
            'Delay': f'{episode_delay:.2f}s',
            'Energy': f'{episode_energy:.2e}J'
        })
        
        # بهترین مدل
        if avg_reward > best_reward:
            best_reward = avg_reward
            no_improvement = 0
            agent.save_models('models/checkpoints/best_model')
            if episode % 50 == 0:  # پرینت هر 50 اپیزود
                print(f"\n✨ بهترین مدل ذخیره شد (Episode {episode}): {best_reward:.2e}")
        else:
            no_improvement += 1
        
        # Checkpoint
        if episode % SAVE_INTERVAL == 0:
            agent.save_models(f'models/checkpoints/episode_{episode}')
            print(f"\n💾 Checkpoint ذخیره شد: Episode {episode}")
        
        # Early stopping
        if no_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️ Early Stopping: {EARLY_STOP_PATIENCE} اپیزود بدون بهبود")
            break
    
    pbar.close()
    
    # ذخیره نتایج
    print("\n" + "="*60)
    print("💾 در حال ذخیره نتایج...")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    results = {
        'timestamp': timestamp,
        'episodes': len(episode_rewards),
        'best_reward': float(best_reward),
        'final_avg_reward': float(np.mean(episode_rewards[-100:])) if len(episode_rewards) >= 100 else float(np.mean(episode_rewards)),
        'rewards': [float(r) for r in episode_rewards],
        'delays': [float(d) for d in episode_delays],
        'energies': [float(e) for e in episode_energies]
    }
    
    json_path = f'results/maddpg_training/training_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   ✓ JSON: {json_path}")
    
    # نمودارها
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes_range = range(1, len(episode_rewards) + 1)
    
    # Reward
    axes[0, 0].plot(episodes_range, episode_rewards, alpha=0.3, label='Episode Reward')
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        axes[0, 0].plot(range(100, len(episode_rewards) + 1), moving_avg, 
                        label='Moving Avg (100)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Delay
    axes[0, 1].plot(episodes_range, episode_delays, alpha=0.3)
    if len(episode_delays) >= 100:
        moving_avg = np.convolve(episode_delays, np.ones(100)/100, mode='valid')
        axes[0, 1].plot(range(100, len(episode_delays) + 1), moving_avg, linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Delay (s)')
    axes[0, 1].set_title('Episode Delay')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy
    axes[1, 0].plot(episodes_range, episode_energies, alpha=0.3)
    if len(episode_energies) >= 100:
        moving_avg = np.convolve(episode_energies, np.ones(100)/100, mode='valid')
        axes[1, 0].plot(range(100, len(episode_energies) + 1), moving_avg, linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Total Energy (J)')
    axes[1, 0].set_title('Episode Energy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Delay vs Energy
    axes[1, 1].scatter(episode_delays, episode_energies, alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Delay (s)')
    axes[1, 1].set_ylabel('Energy (J)')
    axes[1, 1].set_title('Delay-Energy Trade-off')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    png_path = f'results/maddpg_training/training_{timestamp}.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ PNG: {png_path}")
    plt.close()
    
    # TXT
    txt_path = f'results/maddpg_training/training_{timestamp}.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("نتایج آموزش MADDPG\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"تاریخ: {timestamp}\n")
        f.write(f"تعداد اپیزودها: {len(episode_rewards)}\n\n")
        f.write(f"بهترین Reward: {best_reward:.2e}\n")
        last_100 = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
        f.write(f"میانگین آخر:\n")
        f.write(f"  - Reward: {np.mean(last_100):.2e}\n")
        f.write(f"  - Delay: {np.mean(episode_delays[-len(last_100):]):.2f} s\n")
        f.write(f"  - Energy: {np.mean(episode_energies[-len(last_100):]):.2e} J\n")
    print(f"   ✓ TXT: {txt_path}")
    
    # مدل نهایی
    agent.save_models('models/maddpg_final')
    print(f"   ✓ Models: models/maddpg_final/")
    
    print("\n" + "="*60)
    print("✅ آموزش با موفقیت به پایان رسید!")
    print("="*60)

if __name__ == '__main__':
    train_maddpg()
