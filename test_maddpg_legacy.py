"""
test_maddpg_legacy.py (نسخه نهایی - کار می‌کند!)
=====================================================
Fix: استخراج state از دیکشنری برای هر عامل
"""

import torch
import torch.nn as nn
import numpy as np
from core.env_multi import MultiUAVEnv
from tqdm import tqdm
import json
import os


class ActorLegacy(nn.Module):
    """معماری Actor Legacy"""
    def __init__(self, state_dim, action_dim, n_agents, hidden_dim=128):
        super(ActorLegacy, self).__init__()
        input_dim = state_dim * 2
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))
        return action


def extract_agent_state(state_dict, agent_idx, state_dim=6):
    """
    استخراج state عامل خاص از دیکشنری محیط
    
    state_dim = 6 شامل:
    [pos_x, pos_y, velocity, angle, energy, distance]
    """
    uav_pos = state_dict['uav_positions'][agent_idx]  # (2,)
    uav_vel = state_dict['uav_velocities'][agent_idx]  # scalar
    uav_angle = state_dict['uav_angles'][agent_idx]    # scalar
    energy = state_dict['energy'][agent_idx]           # scalar
    distance = state_dict['distances'][agent_idx]      # scalar
    
    # ترکیب به یک بردار 6 بعدی
    agent_state = np.concatenate([
        uav_pos,                    # [0:2]
        [uav_vel],                  # [2]
        [uav_angle],                # [3]
        [energy],                   # [4]
        [distance]                  # [5]
    ])
    
    return agent_state


def prepare_legacy_state(state, state_dim=6):
    """تبدیل state به فرمت (1 x 12)"""
    if isinstance(state, np.ndarray):
        state = torch.FloatTensor(state)
    
    if state.dim() == 1:
        state = state.unsqueeze(0)
    
    # دو برابر کردن: (1 x 6) -> (1 x 12)
    state_doubled = torch.cat([state, state], dim=-1)
    return state_doubled


def test_trained_model_legacy(num_episodes=100, save_results=True):
    """ارزیابی مدل با معماری Legacy"""
    print("="*70)
    print("🧪 ارزیابی مدل MADDPG (Legacy Architecture)")
    print("="*70)
    
    # 1. محیط
    n_agents = 3
    n_users = 5
    env = MultiUAVEnv(n_agents=n_agents, n_users=n_users)
    print(f"✓ محیط بارگذاری شد (Agents={n_agents}, Users={n_users})")
    
    # 2. Actor
    state_dim = 6
    action_dim = 4
    hidden_dim = 128
    
    actor = ActorLegacy(state_dim, action_dim, n_agents, hidden_dim)
    print("✓ Actor Legacy ساخته شد")
    
    # 3. بارگذاری وزن‌ها
    device = torch.device("cpu")
    actor_path = 'models/actor_agent0.pt'
    
    if not os.path.exists(actor_path):
        print(f"❌ خطا: فایل {actor_path} پیدا نشد!")
        return None
    
    try:
        state_dict = torch.load(actor_path, map_location=device)
        actor.load_state_dict(state_dict)
        actor.eval()
        print(f"✅ Actor بارگذاری شد از {actor_path}")
    except Exception as e:
        print(f"❌ خطا در بارگذاری: {e}")
        return None
    
    print("\n🎯 شروع تست...")
    print("-"*70)
    
    # 4. ذخیره نتایج
    test_rewards = []
    test_energies = []
    test_delays = []
    episode_details = []
    
    # 5. حلقه تست
    for ep in tqdm(range(num_episodes), desc="Testing"):
        state_dict = env.reset()  # دیکشنری است!
        
        episode_reward = 0
        episode_energy = 0
        episode_delay = 0
        step_count = 0
        
        for step in range(200):
            with torch.no_grad():
                actions = []
                
                # 🔧 استخراج state از دیکشنری برای هر عامل
                for i in range(n_agents):
                    agent_state = extract_agent_state(state_dict, i, state_dim)
                    state_legacy = prepare_legacy_state(agent_state, state_dim)
                    action = actor(state_legacy).cpu().numpy()
                    
                    if action.ndim > 1:
                        action = action[0]
                    
                    actions.append(action)
                
                actions = np.array(actions)
            
            # اجرای اکشن
            next_state_dict, reward, done, info = env.step(actions)
            
            # 🔧 reward ممکن است آرایه باشد
            if isinstance(reward, np.ndarray):
                reward = reward.sum()
            
            episode_reward += reward
            episode_energy += info.get('energy_total', 0)
            episode_delay += info.get('mean_delay', 0)
            
            state_dict = next_state_dict
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
    
    # 6. خلاصه نتایج
    print("\n" + "="*70)
    print("📊 نتایج نهایی")
    print("="*70)
    
    mean_reward = np.mean(test_rewards)
    mean_energy = np.mean(test_energies)
    mean_delay = np.mean(test_delays)
    
    print(f"📈 Mean Reward:  {mean_reward:>12.4f}  (±{np.std(test_rewards):.4f})")
    print(f"⚡ Mean Energy:  {mean_energy:>12.4f}  (±{np.std(test_energies):.4f})")
    print(f"⏱️  Mean Delay:   {mean_delay:>12.4f}  (±{np.std(test_delays):.4f})")
    
    # 7. ذخیره نتایج
    if save_results:
        results = {
            'summary': {
                'num_episodes': num_episodes,
                'mean_reward': float(mean_reward),
                'std_reward': float(np.std(test_rewards)),
                'mean_energy': float(mean_energy),
                'std_energy': float(np.std(test_energies)),
                'mean_delay': float(mean_delay),
                'std_delay': float(np.std(test_delays))
            },
            'episodes': episode_details
        }
        
        # JSON
        json_path = 'results/test_results_legacy.json'
        os.makedirs('results', exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 نتایج JSON ذخیره شد: {json_path}")
        
        # NPZ
        npz_path = 'results/test_results_legacy.npz'
        np.savez(
            npz_path,
            rewards=test_rewards,
            energies=test_energies,
            delays=test_delays
        )
        print(f"💾 نتایج NPZ ذخیره شد: {npz_path}")
    
    print("\n✅ تست با موفقیت به پایان رسید!")
    return results


if __name__ == "__main__":
    results = test_trained_model_legacy(num_episodes=100, save_results=True)
