"""
run_obstacle_experiments_hybrid.py
===================================
آزمایش سیاست‌های مختلف شامل Hybrid Policy
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.env_multi import MultiUAVEnv

# ==================== POLICIES ====================

class RandomPolicy:
    """سیاست تصادفی - Baseline"""
    def __init__(self, n_agents=3):
        self.n_agents = n_agents
    
    def select_action(self, state):
        actions = []
        for _ in range(self.n_agents):
            v = np.random.uniform(10.0, 25.0)      # سرعت
            theta = np.random.uniform(0, 2*np.pi)  # زاویه
            f = np.random.uniform(1e9, 3e9)        # فرکانس CPU
            o = np.random.uniform(0.3, 0.9)        # نسبت offload
            actions.append(np.array([v, theta, f, o], dtype=np.float32))
        return actions


class GreedyPolicy:
    """سیاست حریصانه - حرکت به سمت نزدیک‌ترین کاربر"""
    def __init__(self, n_agents=3):
        self.n_agents = n_agents
    
    def select_action(self, state):
        # استخراج موقعیت‌ها
        uav_positions = state['uav_positions']  # shape: (n_agents, 2)
        user_positions = state.get('user_positions', np.array([[50, 50]]))
        
        actions = []
        for i in range(self.n_agents):
            uav_pos = uav_positions[i]
            
            # یافتن نزدیک‌ترین کاربر
            distances = np.linalg.norm(user_positions - uav_pos, axis=1)
            nearest_user = user_positions[np.argmin(distances)]
            
            # محاسبه زاویه حرکت
            direction = nearest_user - uav_pos
            theta = np.arctan2(direction[1], direction[0])
            
            # پارامترهای ثابت بهینه
            v = 20.0      # سرعت متوسط
            f = 2.0e9     # فرکانس CPU متوسط
            o = 0.7       # 70% offload
            
            actions.append(np.array([v, theta, f, o], dtype=np.float32))
        
        return actions


class ObstacleAwarePolicy:
    """سیاست آگاه از موانع - متعادل‌تر برای پایداری"""
    def __init__(self, n_agents=3):
        self.n_agents = n_agents
    
    def select_action(self, state):
        uav_positions = state['uav_positions']
        user_positions = state.get('user_positions', np.array([[50, 50]]))
        
        actions = []
        for i in range(self.n_agents):
            uav_pos = uav_positions[i]
            
            # یافتن نزدیک‌ترین کاربر
            distances = np.linalg.norm(user_positions - uav_pos, axis=1)
            nearest_user = user_positions[np.argmin(distances)]
            
            # محاسبه زاویه
            direction = nearest_user - uav_pos
            theta = np.arctan2(direction[1], direction[0])
            
            # پارامترهای محافظه‌کارانه (برای کاهش delay)
            v = 15.0      # سرعت کمتر برای مانور بهتر
            f = 2.2e9     # فرکانس کمی بالاتر
            o = 0.5       # توازن 50-50
            
            actions.append(np.array([v, theta, f, o], dtype=np.float32))
        
        return actions


class HybridPolicy:
    """
    🌟 سیاست ترکیبی هوشمند
    ========================
    ترکیب بهترین ویژگی‌های Greedy (reward بالا) و Obstacle-Aware (delay کم)
    
    استراتژی:
    - از ناوبری Greedy استفاده می‌کند (حرکت به سمت نزدیک‌ترین کاربر)
    - پارامترها را به صورت دینامیک بر اساس وضعیت تنظیم می‌کند
    - هدف: بالانس بین Reward، Delay و Energy
    """
    
    def __init__(self, n_agents=3):
        self.n_agents = n_agents
        self.greedy_policy = GreedyPolicy(n_agents)
        
        # پارامترهای قابل تنظیم
        self.base_velocity = 17.5      # بین Greedy (20) و Obstacle-Aware (15)
        self.base_frequency = 2.3e9    # کمی بالاتر برای کاهش delay
        self.base_offload = 0.65       # بین 0.7 و 0.5
    
    def select_action(self, state):
        # ابتدا از Greedy برای محاسبه جهت حرکت استفاده می‌کنیم
        base_actions = self.greedy_policy.select_action(state)
        
        # استخراج اطلاعات اضافی برای تصمیم‌گیری هوشمند
        mean_delay = state.get('mean_delay', 3.0)
        energy_total = state.get('energy_total', 2e4)
        
        # بهینه‌سازی پارامترها بر اساس شرایط
        actions = []
        for i, base_action in enumerate(base_actions):
            theta = base_action[1]  # حفظ زاویه از Greedy
            
            # 🎯 تنظیم دینامیک سرعت
            if mean_delay > 4.0:
                v = 15.0  # سرعت کم برای مانور بهتر
            elif mean_delay < 2.5:
                v = 20.0  # سرعت بالا اگر delay خیلی کم است
            else:
                v = self.base_velocity
            
            # ⚡ تنظیم دینامیک فرکانس
            if energy_total > 3e4:
                f = 2.0e9  # کاهش برای صرفه‌جویی انرژی
            elif energy_total < 1.8e4:
                f = 2.5e9  # افزایش برای کاهش delay بیشتر
            else:
                f = self.base_frequency
            
            # 🔄 تنظیم دینامیک offload
            if mean_delay > 4.0:
                o = 0.5   # کاهش offload
            elif mean_delay < 2.5:
                o = 0.75  # افزایش offload
            else:
                o = self.base_offload
            
            actions.append(np.array([v, theta, f, o], dtype=np.float32))
        
        return actions


# ==================== EXPERIMENT RUNNER ====================

def run_single_experiment(env, policy, n_episodes=50, max_steps=50):
    """اجرای یک آزمایش با یک سیاست"""
    
    total_rewards = []
    total_delays = []
    total_energies = []
    
    for ep in range(n_episodes):
        # 🔧 FIX: reset() فقط یک مقدار برمی‌گرداند
        state = env.reset()
        ep_reward = 0
        ep_steps = 0
        
        for step in range(max_steps):
            # انتخاب action
            actions = policy.select_action(state)
            
            # اجرای action
            step_result = env.step(actions)
            
            # 🔧 FIX: بررسی تعداد مقادیر برگشتی
            if len(step_result) == 5:
                next_state, rewards, dones, truncated, info = step_result
            elif len(step_result) == 4:
                next_state, rewards, dones, info = step_result
                truncated = False
            else:
                raise ValueError(f"Unexpected step() return: {len(step_result)} values")
            
            # جمع reward
            if isinstance(rewards, dict):
                ep_reward += sum(rewards.values())
            elif isinstance(rewards, (list, np.ndarray)):
                ep_reward += sum(rewards)
            else:
                ep_reward += rewards
            
            state = next_state
            ep_steps += 1
            
            # بررسی پایان episode
            if isinstance(dones, dict):
                if all(dones.values()):
                    break
            elif isinstance(dones, (list, np.ndarray)):
                if all(dones):
                    break
            elif dones:
                break
        
        # ثبت نتایج
        total_rewards.append(ep_reward)
        
        # استخراج delay و energy از state نهایی
        delay = state.get('mean_delay', 0)
        energy = state.get('energy_total', 0)
        
        total_delays.append(delay)
        total_energies.append(energy)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes} - "
                  f"Reward: {ep_reward:.2e}, "
                  f"Delay: {delay:.2f}, "
                  f"Energy: {energy:.2e}")
    
    return {
        'rewards': total_rewards,
        'delays': total_delays,
        'energies': total_energies,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_delay': np.mean(total_delays),
        'std_delay': np.std(total_delays),
        'mean_energy': np.mean(total_energies),
        'std_energy': np.std(total_energies)
    }


def main():
    print("="*70)
    print("🚀 آزمایش سیاست‌های مختلف شامل Hybrid Policy")
    print("="*70)
    
    # تنظیمات آزمایش
    n_uavs = 3
    n_episodes = 50
    max_steps = 50
    
    # ایجاد محیط
    print(f"\n📦 ایجاد محیط با {n_uavs} UAV...")
    env = MultiUAVEnv(n_agents=n_uavs)
    
    # تعریف سیاست‌ها
    policies = {
        'Random': RandomPolicy(n_uavs),
        'Greedy': GreedyPolicy(n_uavs),
        'Obstacle-Aware': ObstacleAwarePolicy(n_uavs),
        'Hybrid': HybridPolicy(n_uavs)  # 🌟 سیاست جدید!
    }
    
    # اجرای آزمایش‌ها
    results = []
    
    for name, policy in policies.items():
        print(f"\n{'='*70}")
        print(f"🧪 آزمایش سیاست: {name}")
        print(f"{'='*70}")
        
        result = run_single_experiment(env, policy, n_episodes, max_steps)
        result['policy'] = name
        results.append(result)
        
        print(f"\n📊 نتایج نهایی {name}:")
        print(f"  Reward:  {result['mean_reward']:.2e} ± {result['std_reward']:.2e}")
        print(f"  Delay:   {result['mean_delay']:.2f} ± {result['std_delay']:.2f}")
        print(f"  Energy:  {result['mean_energy']:.2e} ± {result['std_energy']:.2e}")
    
    # مقایسه نهایی
    print("\n" + "="*70)
    print("📈 مقایسه نهایی همه سیاست‌ها")
    print("="*70)
    print(f"{'Policy':<20} {'Reward':>15} {'Delay':>10} {'Energy':>15}")
    print("-"*70)
    
    for r in results:
        print(f"{r['policy']:<20} {r['mean_reward']:>15.2e} "
              f"{r['mean_delay']:>10.2f} {r['mean_energy']:>15.2e}")
    
    # یافتن بهترین‌ها
    print("\n🏆 برندگان در هر معیار:")
    best_reward = max(results, key=lambda x: x['mean_reward'])
    best_delay = min(results, key=lambda x: x['mean_delay'])
    best_energy = min(results, key=lambda x: x['mean_energy'])
    
    print(f"  Reward:  {best_reward['policy']} ({best_reward['mean_reward']:.2e})")
    print(f"  Delay:   {best_delay['policy']} ({best_delay['mean_delay']:.2f})")
    print(f"  Energy:  {best_energy['policy']} ({best_energy['mean_energy']:.2e})")
    
    # بررسی عملکرد Hybrid
    hybrid_result = next(r for r in results if r['policy'] == 'Hybrid')
    greedy_result = next(r for r in results if r['policy'] == 'Greedy')
    obstacle_result = next(r for r in results if r['policy'] == 'Obstacle-Aware')
    
    print("\n🌟 تحلیل عملکرد Hybrid:")
    
    reward_improvement = ((hybrid_result['mean_reward'] - greedy_result['mean_reward']) 
                         / greedy_result['mean_reward'] * 100)
    delay_improvement = ((obstacle_result['mean_delay'] - hybrid_result['mean_delay']) 
                        / obstacle_result['mean_delay'] * 100)
    energy_vs_greedy = ((hybrid_result['mean_energy'] - greedy_result['mean_energy']) 
                       / greedy_result['mean_energy'] * 100)
    
    print(f"  Reward vs Greedy:        {reward_improvement:+.1f}%")
    print(f"  Delay vs Obstacle-Aware: {delay_improvement:+.1f}%")
    print(f"  Energy vs Greedy:        {energy_vs_greedy:+.1f}%")
    
    return results


if __name__ == "__main__":
    results = main()
    print("\n✅ آزمایش با موفقیت پایان یافت!")
