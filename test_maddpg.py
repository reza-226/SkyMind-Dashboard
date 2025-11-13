"""
test_maddpg.py
==============
ارزیابی مدل آموزش‌دیده MADDPG با تبدیل state_dict
"""

import torch
import numpy as np
from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent
from tqdm import tqdm
import json
import os

def convert_old_state_dict(old_state_dict):
    """
    تبدیل state_dict قدیمی (fc1, fc2, fc3) به فرمت جدید (net.0, net.2, net.4)
    
    قدیمی:
        fc1.weight, fc1.bias
        fc2.weight, fc2.bias
        fc3.weight, fc3.bias
    
    جدید:
        net.0.weight, net.0.bias
        net.2.weight, net.2.bias
        net.4.weight, net.4.bias
    """
    new_state_dict = {}
    mapping = {
        'fc1': 'net.0',
        'fc2': 'net.2',
        'fc3': 'net.4'
    }
    
    for old_key, value in old_state_dict.items():
        # مثلاً: fc1.weight -> net.0.weight
        for old_name, new_name in mapping.items():
            if old_key.startswith(old_name):
                new_key = old_key.replace(old_name, new_name)
                new_state_dict[new_key] = value
                break
    
    return new_state_dict


def test_trained_model(num_episodes=100, save_results=True):
    """
    ارزیابی مدل آموزش دیده بدون exploration
    """
    print("="*70)
    print("🧪 ارزیابی مدل MADDPG آموزش‌دیده")
    print("="*70)
    
    # 1. بارگذاری محیط
    n_agents = 3
    n_users = 5
    env = MultiUAVEnv(n_agents=n_agents, n_users=n_users)
    print(f"✓ محیط بارگذاری شد (Agents={n_agents}, Users={n_users})")
    
    # 2. ساخت Agent
    state_dim = 6
    action_dim = 4
    
    agent = MADDPG_Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        lr=1e-4,
        gamma=0.99,
        tau=0.01,
        device='cpu'
    )
    print("✓ Agent ساخته شد")
    
    # 3. بارگذاری با تبدیل state_dict
    device = torch.device("cpu")
    actor_path = 'models/actor_agent0.pt'
    
    if not os.path.exists(actor_path):
        print(f"❌ خطا: فایل {actor_path} پیدا نشد!")
        return None
    
    try:
        # بارگذاری state_dict قدیمی
        old_state_dict = torch.load(actor_path, map_location=device)
        
        print("📋 کلیدهای مدل ذخیره شده:")
        for key in old_state_dict.keys():
            print(f"  - {key}")
        
        # بررسی نوع معماری
        if 'fc1.weight' in old_state_dict:
            print("\n🔧 تبدیل معماری قدیمی به جدید...")
            new_state_dict = convert_old_state_dict(old_state_dict)
            agent.actor.load_state_dict(new_state_dict)
        else:
            # معماری جدید است
            agent.actor.load_state_dict(old_state_dict)
        
        agent.actor.eval()
        print(f"✓ Actor بارگذاری شد از {actor_path}")
        
    except Exception as e:
        print(f"❌ خطا در بارگذاری Actor: {e}")
        return None
    
    print("\n🎯 شروع تست...")
    print("-"*70)
    
    # 4. ذخیره‌سازی نتایج
    test_rewards = []
    test_energies = []
    test_delays = []
    episode_details = []
    
    # 5. حلقه تست
    for ep in tqdm(range(num_episodes), desc="Testing"):
        state = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_delay = 0
        step_count = 0
        
        for step in range(200):
            # دریافت اکشن بدون noise
            with torch.no_grad():
                actions = []
                for i in range(n_agents):
                    action = agent.act(state[i], noise_scale=0.0)
                    actions.append(action)
                actions = np.array(actions)
            
            # اجرای اکشن
            next_state, reward, done, info = env.step(actions)
            
            # جمع‌آوری metrics
            episode_reward += reward
            episode_energy += info.get('energy_total', 0)
            episode_delay += info.get('mean_delay', 0)
            
            state = next_state
            step_count = step + 1
            
            if done:
                break
        
        # ذخیره نتایج
        test_rewards.append(episode_reward)
        test_energies.append(episode_energy)
        test_delays.append(episode_delay)
        
        episode_details.append({
            'episode': ep + 1,
            'reward': float(episode_reward),
            'energy': float(episode_energy),
            'delay': float(episode_delay),
            'steps': step_count
        })
        
        # نمایش هر 10 اپیزود
        if (ep + 1) % 10 == 0:
            print(f"\nEpisode {ep+1:3d}/{num_episodes} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Energy: {episode_energy:.2e} | "
                  f"Delay: {episode_delay:.6f}")
    
    # 6. محاسبه آماره‌ها
    results = {
        'rewards': test_rewards,
        'energies': test_energies,
        'delays': test_delays,
        'episode_details': episode_details,
        'statistics': {
            'reward_mean': float(np.mean(test_rewards)),
            'reward_std': float(np.std(test_rewards)),
            'reward_max': float(np.max(test_rewards)),
            'reward_min': float(np.min(test_rewards)),
            'energy_mean': float(np.mean(test_energies)),
            'energy_std': float(np.std(test_energies)),
            'delay_mean': float(np.mean(test_delays)),
            'delay_std': float(np.std(test_delays))
        }
    }
    
    # 7. نمایش نتایج
    print("\n" + "="*70)
    print("📊 نتایج نهایی ارزیابی")
    print("="*70)
    print(f"{'Metric':<20} {'Mean':<15} {'Std':<15} {'Min/Max':<20}")
    print("-"*70)
    print(f"{'Reward':<20} {results['statistics']['reward_mean']:>10.2f}    "
          f"{results['statistics']['reward_std']:>10.2f}    "
          f"{results['statistics']['reward_min']:>7.2f} / {results['statistics']['reward_max']:<7.2f}")
    print(f"{'Energy':<20} {results['statistics']['energy_mean']:>10.2e}    "
          f"{results['statistics']['energy_std']:>10.2e}")
    print(f"{'Delay':<20} {results['statistics']['delay_mean']:>10.6f}    "
          f"{results['statistics']['delay_std']:>10.6f}")
    print("="*70)
    
    # 8. ذخیره نتایج
    if save_results:
        os.makedirs('results', exist_ok=True)
        
        np.savez('results/test_results.npz',
                 rewards=test_rewards,
                 energies=test_energies,
                 delays=test_delays)
        print("✅ فایل NumPy ذخیره شد: results/test_results.npz")
        
        with open('results/test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("✅ فایل JSON ذخیره شد: results/test_results.json")
    
    return results


if __name__ == "__main__":
    results = test_trained_model(num_episodes=100)
    
    if results is not None:
        print("\n✅ تست با موفقیت انجام شد!")
        print(f"📁 نتایج در پوشه results/ ذخیره شدند")
    else:
        print("\n❌ تست با خطا مواجه شد!")
