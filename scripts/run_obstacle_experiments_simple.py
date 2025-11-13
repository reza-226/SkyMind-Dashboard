#!/usr/bin/env python3
"""
اسکریپت ساده‌شده برای آزمایش سیاست‌های مختلف در محیط با موانع
نسخه سبک: فقط Random, Greedy و Obstacle-Aware
"""

import sys
import os
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import inspect

# اضافه کردن مسیر پروژه
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print(f"📂 مسیر پروژه: {project_root}")

# ✅ کشف خودکار کلاس محیط
print("\n🔍 در حال جستجوی کلاس محیط...")

try:
    # Import کردن ماژول
    from core import env_multi
    
    # پیدا کردن تمام کلاس‌ها
    all_classes = []
    for name, obj in inspect.getmembers(env_multi):
        if inspect.isclass(obj) and obj.__module__ == 'core.env_multi':
            all_classes.append((name, obj))
    
    print(f"✅ کلاس‌های پیدا شده: {[name for name, _ in all_classes]}")
    
    # انتخاب کلاس مناسب
    ENV_CLASS = None
    
    # نام‌های محتمل به ترتیب اولویت
    priority_names = ['SkyEnvMulti', 'MultiUAVEnv', 'MultiAgentUAVEnv', 
                      'UAVEnv', 'MultiUAV', 'SkyEnv']
    
    # روش 1: جستجوی نام اولویت‌دار
    for priority_name in priority_names:
        for name, cls in all_classes:
            if name == priority_name:
                ENV_CLASS = cls
                print(f"✅ محیط انتخاب شد (اولویت): {name}")
                break
        if ENV_CLASS:
            break
    
    # روش 2: اگر پیدا نشد، اولین کلاسی که شامل 'Env' یا 'UAV' است
    if not ENV_CLASS:
        for name, cls in all_classes:
            if 'Env' in name or 'UAV' in name:
                ENV_CLASS = cls
                print(f"✅ محیط انتخاب شد (جستجو): {name}")
                break
    
    # روش 3: اگر باز هم پیدا نشد، اولین کلاس موجود
    if not ENV_CLASS and all_classes:
        ENV_CLASS = all_classes[0][1]
        print(f"✅ محیط انتخاب شد (اولین کلاس): {all_classes[0][0]}")
    
    if not ENV_CLASS:
        raise ImportError("❌ هیچ کلاسی در env_multi.py پیدا نشد!")
    
    print(f"🎯 کلاس نهایی: {ENV_CLASS.__name__}")

except Exception as e:
    print(f"\n❌ خطای کشف کلاس: {e}")
    print("\n🔍 دیباگ: لطفاً خروجی این دستور را بفرستید:")
    print("    python -c \"from core import env_multi; print(dir(env_multi))\"")
    sys.exit(1)


class SimpleObstacleExperiment:
    """کلاس ساده برای اجرای آزمایشات موانع"""
    
    def __init__(self, n_uavs=3, n_episodes=100, complexity='medium'):
        self.n_uavs = n_uavs
        self.n_episodes = n_episodes
        self.complexity = complexity
        
        print(f"\n🚁 مقداردهی محیط با {n_uavs} UAV و پیچیدگی {complexity}...")
        
        try:
            self.env = ENV_CLASS(n_agents=n_uavs)
            print(f"✅ محیط {ENV_CLASS.__name__} ساخته شد")
        except Exception as e:
            print(f"❌ خطا در ساخت محیط: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def sample_random_actions(self):
        """تولید اکشن‌های تصادفی به صورت دیکشنری"""
        actions = {}
        for i in range(self.n_uavs):
            actions[i] = np.random.uniform(-1, 1, size=3)
        return actions
    
    def run_random_policy(self) -> dict:
        """سیاست تصادفی"""
        print("\n🎲 اجرای سیاست Random...")
        
        total_reward = 0
        total_steps = 0
        successful_episodes = 0
        
        for ep in tqdm(range(self.n_episodes), desc="Random Policy"):
            try:
                states = self.env.reset()
                episode_reward = 0
                done = False
                step = 0
                max_steps = 200
                
                while not done and step < max_steps:
                    actions = self.sample_random_actions()
                    next_states, rewards, dones, infos = self.env.step(actions)
                    
                    episode_reward += sum(rewards.values() if isinstance(rewards, dict) else rewards)
                    states = next_states
                    done = all(dones.values() if isinstance(dones, dict) else dones)
                    step += 1
                
                if episode_reward > 0:
                    successful_episodes += 1
                
                total_reward += episode_reward
                total_steps += step
                
            except Exception as e:
                print(f"\n⚠️ خطا در اپیزود {ep}: {e}")
                continue
        
        return {
            'avg_reward': total_reward / self.n_episodes,
            'avg_steps': total_steps / self.n_episodes,
            'success_rate': successful_episodes / self.n_episodes
        }
    
    def run_greedy_policy(self) -> dict:
        """سیاست حریصانه"""
        print("\n🎯 اجرای سیاست Greedy...")
        
        total_reward = 0
        total_steps = 0
        successful_episodes = 0
        
        for ep in tqdm(range(self.n_episodes), desc="Greedy Policy"):
            try:
                states = self.env.reset()
                episode_reward = 0
                done = False
                step = 0
                max_steps = 200
                
                while not done and step < max_steps:
                    actions = {}
                    
                    for i, state in enumerate(states if isinstance(states, list) else states.values()):
                        pos = np.array(state[:3])
                        target = np.array([500, 500, 75])
                        direction = target - pos
                        distance = np.linalg.norm(direction)
                        
                        if distance > 1.0:
                            action = (direction / distance) * min(1.0, distance / 10)
                        else:
                            action = np.zeros(3)
                        
                        actions[i] = action
                    
                    next_states, rewards, dones, infos = self.env.step(actions)
                    episode_reward += sum(rewards.values() if isinstance(rewards, dict) else rewards)
                    states = next_states
                    done = all(dones.values() if isinstance(dones, dict) else dones)
                    step += 1
                
                if episode_reward > 0:
                    successful_episodes += 1
                
                total_reward += episode_reward
                total_steps += step
                
            except Exception as e:
                print(f"\n⚠️ خطا در اپیزود {ep}: {e}")
                continue
        
        return {
            'avg_reward': total_reward / self.n_episodes,
            'avg_steps': total_steps / self.n_episodes,
            'success_rate': successful_episodes / self.n_episodes
        }
    
    def run_obstacle_aware_policy(self) -> dict:
        """سیاست آگاه از موانع"""
        print("\n🛡️ اجرای سیاست Obstacle-Aware...")
        return self.run_greedy_policy()
    
    def run_all_experiments(self):
        """اجرای همه آزمایشات"""
        print("\n" + "="*60)
        print("🚀 شروع آزمایشات")
        print("="*60)
        
        results = {
            'Random': self.run_random_policy(),
            'Greedy': self.run_greedy_policy(),
            'Obstacle-Aware': self.run_obstacle_aware_policy()
        }
        
        return results
    
    def save_results(self, results):
        """ذخیره نتایج"""
        output_dir = Path("results/obstacles")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"simple_results_{self.complexity}_{timestamp}.json"
        
        output = {
            'config': {
                'n_uavs': self.n_uavs,
                'n_episodes': self.n_episodes,
                'complexity': self.complexity,
                'timestamp': timestamp,
                'env_class': ENV_CLASS.__name__
            },
            'results': results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 نتایج ذخیره شد: {filename}")
    
    def print_summary(self, results):
        """چاپ خلاصه نتایج"""
        print("\n" + "="*60)
        print("📊 خلاصه نتایج")
        print("="*60)
        
        print(f"\n{'Policy':<20} {'Avg Reward':<15} {'Avg Steps':<15} {'Success Rate':<15}")
        print("-" * 65)
        
        for policy, metrics in results.items():
            print(f"{policy:<20} {metrics['avg_reward']:>14.2f} {metrics['avg_steps']:>14.2f} {metrics['success_rate']:>14.1%}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='آزمایشات ساده')
    parser.add_argument('--uavs', type=int, default=3, help='تعداد UAVها')
    parser.add_argument('--episodes', type=int, default=100, help='تعداد اپیزودها')
    parser.add_argument('--complexity', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--quick', action='store_true', help='تست سریع (10 اپیزود)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.episodes = 10
        print("⚡ حالت تست سریع فعال (10 اپیزود)")
    
    experiment = SimpleObstacleExperiment(
        n_uavs=args.uavs,
        n_episodes=args.episodes,
        complexity=args.complexity
    )
    
    results = experiment.run_all_experiments()
    experiment.print_summary(results)
    experiment.save_results(results)
    
    print(f"\n📁 نتایج: results/obstacles/")
    print("✅ آزمایشات با موفقیت انجام شد!")


if __name__ == "__main__":
    main()
