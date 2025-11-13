# test_action_format.py
import sys
sys.path.insert(0, 'D:/Payannameh/SkyMind-Dashboard')

from core.env_multi import MultiUAVEnv
import numpy as np

env = MultiUAVEnv(n_agents=3)
obs = env.reset()

print("=" * 60)
print("🎮 تست دقیق Action Format")
print("=" * 60)

# تست شکل‌های مختلف
test_formats = [
    ("(3,)", np.random.rand(3)),
    ("(3, 1)", np.random.rand(3, 1)),
    ("(3, 2)", np.random.rand(3, 2)),
    ("(3, 3)", np.random.rand(3, 3)),
]

for name, action in test_formats:
    try:
        env.reset()
        next_obs, reward, done, info = env.step(action)
        print(f"✅ {name}: موفق!")
        print(f"   reward: {reward}")
        print(f"   done: {done}")
        print(f"   action shape: {action.shape}")
        break
    except Exception as e:
        print(f"❌ {name}: {str(e)[:80]}")

print("=" * 60)
