"""
run_obstacle_experiments_FINAL_FIXED.py
=======================================
نسخه نهایی با استخراج صحیح Energy و Delay
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.env_multi import MultiUAVEnv
import numpy as np
import argparse
import time
from pathlib import Path
import json

class BasePolicy:
    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents
    
    def select_action(self, state, step_count, avg_delay, avg_energy):
        raise NotImplementedError

class RandomPolicy(BasePolicy):
    def select_action(self, state, step_count=0, avg_delay=0, avg_energy=0):
        actions = []
        for i in range(self.n_agents):
            v = np.random.uniform(5, 30)
            theta = np.random.uniform(0, 2*np.pi)
            f = np.random.uniform(1e9, 3e9)
            o = np.random.uniform(0.3, 1.0)
            actions.append(np.array([v, theta, f, o], dtype=np.float32))
        return actions

class GreedyPolicy(BasePolicy):
    def select_action(self, state, step_count=0, avg_delay=0, avg_energy=0):
        uav_positions = state['uav_positions']
        user_positions = state['user_positions']
        
        actions = []
        for i in range(self.n_agents):
            uav_pos = uav_positions[i]
            distances = np.linalg.norm(user_positions - uav_pos, axis=1)
            closest_user_idx = np.argmin(distances)
            target = user_positions[closest_user_idx]
            
            delta = target - uav_pos
            theta = np.arctan2(delta[1], delta[0])
            v = 25.0
            f = 2.5e9
            o = 0.8
            
            actions.append(np.array([v, theta, f, o], dtype=np.float32))
        return actions

class ObstacleAwarePolicy(BasePolicy):
    def select_action(self, state, step_count=0, avg_delay=0, avg_energy=0):
        uav_positions = state['uav_positions']
        user_positions = state['user_positions']
        
        actions = []
        for i in range(self.n_agents):
            uav_pos = uav_positions[i]
            
            # جلوگیری از برخورد با UAVهای دیگر
            safe_angle = 0
            min_safe_distance = 50
            for j in range(self.n_agents):
                if i != j:
                    other_pos = uav_positions[j]
                    dist = np.linalg.norm(other_pos - uav_pos)
                    if dist < min_safe_distance:
                        repel_vector = uav_pos - other_pos
                        safe_angle += np.arctan2(repel_vector[1], repel_vector[0])
            
            # حرکت به سمت کاربران
            distances = np.linalg.norm(user_positions - uav_pos, axis=1)
            closest_user_idx = np.argmin(distances)
            target = user_positions[closest_user_idx]
            delta = target - uav_pos
            target_angle = np.arctan2(delta[1], delta[0])
            
            # ترکیب زاویه هدف و زاویه امن
            theta = (target_angle + safe_angle) / 2
            v = 20.0
            f = 2e9
            o = 0.9
            
            actions.append(np.array([v, theta, f, o], dtype=np.float32))
        return actions

class HybridPolicy(BasePolicy):
    def select_action(self, state, step_count=0, avg_delay=0, avg_energy=0):
        uav_positions = state['uav_positions']
        user_positions = state['user_positions']
        
        # تنظیم دینامیک پارامترها بر اساس وضعیت فعلی
        if avg_delay > 5.0:  # Delay زیاد
            v_base, f_base, o_base = 28.0, 2.8e9, 0.7
        elif avg_energy > 50000:  # مصرف انرژی زیاد
            v_base, f_base, o_base = 15.0, 1.8e9, 0.85
        else:  # حالت متعادل
            v_base, f_base, o_base = 22.0, 2.3e9, 0.8
        
        actions = []
        for i in range(self.n_agents):
            uav_pos = uav_positions[i]
            
            # Obstacle avoidance
            safe_angle = 0
            min_safe_distance = 60
            for j in range(self.n_agents):
                if i != j:
                    other_pos = uav_positions[j]
                    dist = np.linalg.norm(other_pos - uav_pos)
                    if dist < min_safe_distance:
                        repel_vector = uav_pos - other_pos
                        safe_angle += np.arctan2(repel_vector[1], repel_vector[0]) * 0.3
            
            # Target selection
            distances = np.linalg.norm(user_positions - uav_pos, axis=1)
            closest_user_idx = np.argmin(distances)
            target = user_positions[closest_user_idx]
            delta = target - uav_pos
            target_angle = np.arctan2(delta[1], delta[0])
            
            theta = target_angle + safe_angle
            
            # تنظیم دقیق پارامترها بر اساس فاصله
            dist_to_target = distances[closest_user_idx]
            if dist_to_target < 100:
                v = v_base * 0.7
                f = f_base * 1.1
            else:
                v = v_base
                f = f_base
            
            actions.append(np.array([v, theta, f, o_base], dtype=np.float32))
        return actions

def run_single_experiment(policy_name, policy, env, n_episodes=50):
    """اجرای یک آزمایش با استخراج صحیح متریک‌ها"""
    
    print(f"\n{'='*70}")
    print(f"🚀 شروع آزمایش: {policy_name}")
    print(f"{'='*70}")
    
    episode_rewards = []
    episode_delays = []
    episode_energies = []
    
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        # لیست‌های جمع‌آوری برای محاسبه میانگین در اپیزود
        step_delays = []
        step_energies = []
        
        while not done and step_count < 100:
            # محاسبه میانگین‌های موقت برای Hybrid
            avg_delay = np.mean(step_delays) if step_delays else 0
            avg_energy = np.mean(step_energies) if step_energies else 0
            
            actions = policy.select_action(state, step_count, avg_delay, avg_energy)
            
            step_result = env.step(actions)
            
            # Unpack با توجه به تعداد خروجی‌ها
            if len(step_result) == 4:
                next_state, rewards, done, _ = step_result
            elif len(step_result) == 5:
                next_state, rewards, done, _, info = step_result
            else:
                raise ValueError(f"خروجی نامعتبر از step(): {len(step_result)} مقدار")
            
            # ✅ استخراج صحیح Energy از state
            current_energies = next_state.get('energy', np.zeros(env.n_agents))
            total_energy_step = np.sum(current_energies)
            
            # ✅ محاسبه Delay از distances (تقسیم بر سرعت متوسط)
            current_distances = next_state.get('distances', np.zeros(env.n_agents))
            current_velocities = next_state.get('uav_velocities', np.ones(env.n_agents) * 20)
            delays = current_distances / (current_velocities + 1e-6)  # جلوگیری از تقسیم بر صفر
            mean_delay_step = np.mean(delays)
            
            # ذخیره در لیست‌ها
            step_delays.append(mean_delay_step)
            step_energies.append(total_energy_step)
            
            total_reward += np.sum(rewards)
            state = next_state
            step_count += 1
        
        # میانگین اپیزود
        episode_rewards.append(total_reward)
        episode_delays.append(np.mean(step_delays))
        episode_energies.append(np.mean(step_energies))
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} | "
                  f"Reward: {total_reward:.2e} | "
                  f"Delay: {episode_delays[-1]:.2f}s | "
                  f"Energy: {episode_energies[-1]:.2e}J")
    
    results = {
        'policy': policy_name,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_delay': float(np.mean(episode_delays)),
        'std_delay': float(np.std(episode_delays)),
        'mean_energy': float(np.mean(episode_energies)),
        'std_energy': float(np.std(episode_energies)),
        'all_rewards': [float(r) for r in episode_rewards],
        'all_delays': [float(d) for d in episode_delays],
        'all_energies': [float(e) for e in episode_energies]
    }
    
    print(f"\n📊 خلاصه نتایج {policy_name}:")
    print(f"  Reward : {results['mean_reward']:.2e} ± {results['std_reward']:.2e}")
    print(f"  Delay  : {results['mean_delay']:.2f}s ± {results['std_delay']:.2f}s")
    print(f"  Energy : {results['mean_energy']:.2e} ± {results['std_energy']:.2e}J")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--output', type=str, default='results/obstacle_experiments_fixed.json')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🔬 آزمایش نهایی با استخراج صحیح Energy و Delay")
    print("="*70)
    print(f"تعداد اپیزودها: {args.episodes}")
    print(f"تعداد UAVها: {args.n_agents}")
    
    env = MultiUAVEnv(n_agents=args.n_agents)
    
    policies = {
        'Random': RandomPolicy(env),
        'Greedy': GreedyPolicy(env),
        'Obstacle-Aware': ObstacleAwarePolicy(env),
        'Hybrid': HybridPolicy(env)
    }
    
    all_results = {}
    
    for name, policy in policies.items():
        start_time = time.time()
        results = run_single_experiment(name, policy, env, args.episodes)
        elapsed = time.time() - start_time
        results['execution_time'] = elapsed
        all_results[name] = results
        print(f"⏱️ زمان اجرا: {elapsed:.2f}s\n")
    
    # ذخیره نتایج
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ نتایج ذخیره شد در: {output_path}")
    
    # جدول مقایسه
    print("\n" + "="*70)
    print("📊 جدول مقایسه نهایی")
    print("="*70)
    print(f"{'Policy':<20} {'Reward':<15} {'Delay (s)':<15} {'Energy (J)':<15}")
    print("-"*70)
    for name, res in all_results.items():
        print(f"{name:<20} {res['mean_reward']:<15.2e} "
              f"{res['mean_delay']:<15.2f} {res['mean_energy']:<15.2e}")

if __name__ == "__main__":
    main()
