# run_obstacle_experiments_final.py
import sys
import os
import time
import numpy as np
import argparse

# اضافه کردن مسیر پروژه
sys.path.insert(0, 'D:/Payannameh/SkyMind-Dashboard')

def discover_env_class():
    """کشف کلاس محیط"""
    from core import env_multi
    classes = [name for name in dir(env_multi)
               if isinstance(getattr(env_multi, name), type)]
    for cls_name in classes:
        cls = getattr(env_multi, cls_name)
        if hasattr(cls, 'reset') and hasattr(cls, 'step'):
            print(f"✅ کلاس محیط پیدا شد: {cls_name}")
            return cls
    raise ImportError("هیچ کلاس محیط مناسب با reset و step یافت نشد!")

ENV_CLASS = discover_env_class()

def create_env(n_uavs=3):
    """ایجاد محیط با پارامتر درست"""
    return ENV_CLASS(n_agents=n_uavs)

# ------------------------------
# سیاست‌های ساده
# ------------------------------
class RandomPolicy:
    def __init__(self, n_agents=3): self.n_agents = n_agents
    def select_action(self, state):
        actions = []
        for _ in range(self.n_agents):
            v = np.random.uniform(1.0, 30.0)
            theta = np.random.uniform(-np.pi, np.pi)
            f = np.random.uniform(0.5e9, 3e9)
            o = np.random.uniform(0.0, 1.0)
            actions.append(np.array([v, theta, f, o]))
        return actions

class GreedyPolicy:
    def __init__(self, n_agents=3): self.n_agents = n_agents
    def select_action(self, state):
        actions = []
        uav_positions = state.get('uav_positions', np.zeros((self.n_agents, 2)))
        user_positions = state.get('user_positions', np.zeros((10, 2)))
        for i in range(self.n_agents):
            uav_pos = uav_positions[i]
            distances = np.linalg.norm(user_positions - uav_pos, axis=1)
            nearest_user = user_positions[np.argmin(distances)]
            direction = nearest_user - uav_pos
            theta = np.arctan2(direction[1], direction[0])
            v, f, o = 20.0, 2.0e9, 0.7
            actions.append(np.array([v, theta, f, o]))
        return actions

class ObstacleAwarePolicy:
    def __init__(self, n_agents=3):
        self.greedy = GreedyPolicy(n_agents)
    def select_action(self, state):
        actions = self.greedy.select_action(state)
        for a in actions:
            a[0], a[2], a[3] = 15.0, 1.5e9, 0.5
        return actions

# ------------------------------
# تابع آزمایش
# ------------------------------
def run_experiment(policy_name, n_uavs=3, episodes=10, max_steps=20):
    print(f"\n{'='*60}")
    print(f"🚀 آزمایش: {policy_name}")
    print(f"{'='*60}")
    env = create_env(n_uavs)
    if policy_name == 'random':
        policy = RandomPolicy(n_uavs)
    elif policy_name == 'greedy':
        policy = GreedyPolicy(n_uavs)
    else:
        policy = ObstacleAwarePolicy(n_uavs)

    total_rewards, total_delays, total_energy = [], [], []
    for ep in range(episodes):
        state = env.reset()
        r_sum, d_sum, e_sum = 0, 0, 0
        for step in range(max_steps):
            actions = policy.select_action(state)
            next_state, reward, done, info = env.step(actions)
            r_sum += np.mean(reward)
            d_sum += info.get('mean_delay', 0)
            e_sum += info.get('energy_total', 0)
            state = next_state
            if done:
                break
        total_rewards.append(r_sum)
        total_delays.append(d_sum)
        total_energy.append(e_sum)
        print(f"Ep {ep+1}/{episodes} | R={r_sum:.2e} D={d_sum:.2e} E={e_sum:.2e}")

    print(f"\n📊 نتایج نهایی برای {policy_name}")
    print(f" Reward میانگین: {np.mean(total_rewards):.2e}")
    print(f" Delay میانگین:  {np.mean(total_delays):.2e}")
    print(f" Energy میانگین: {np.mean(total_energy):.2e}")

    return {
        'policy': policy_name,
        'mean_reward': np.mean(total_rewards),
        'mean_delay': np.mean(total_delays),
        'mean_energy': np.mean(total_energy)
    }

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_uavs', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    if args.quick:
        args.episodes, args.max_steps = 5, 10
        print("⚡ تست سریع فعال شد!")

    # اجرای سه سیاست پایه
    results = []
    for pol in ['random', 'greedy', 'obstacle_aware']:
        results.append(run_experiment(pol, args.n_uavs, args.episodes, args.max_steps))

    print(f"\n{'='*60}")
    print("🏆 خلاصه نهایی آزمایش‌ها")
    for r in results:
        print(f"{r['policy']:>15} | Reward={r['mean_reward']:.2e}, Delay={r['mean_delay']:.2e}, Energy={r['mean_energy']:.2e}")
    print("="*60)
