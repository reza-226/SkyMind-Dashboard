# test_correct_action.py
import sys
sys.path.insert(0, 'D:/Payannameh/SkyMind-Dashboard')

from core.env_multi import MultiUAVEnv
import numpy as np

env = MultiUAVEnv(n_agents=3)
obs = env.reset()

print("=" * 60)
print("🎮 تست Action با فرمت صحیح")
print("=" * 60)

# فرمت صحیح: لیست از آرایه‌های 4 عنصری
actions = [
    np.array([15.0, 0.5, 1.5e9, 0.5]),  # UAV 0
    np.array([15.0, 0.5, 1.5e9, 0.5]),  # UAV 1
    np.array([15.0, 0.5, 1.5e9, 0.5]),  # UAV 2
]

print(f"\n📋 Action format:")
print(f"   Type: {type(actions)}")
print(f"   Length: {len(actions)}")
print(f"   Sample: {actions[0]}")

try:
    next_obs, reward, done, info = env.step(actions)
    print(f"\n✅ موفق!")
    print(f"   Reward: {reward}")
    print(f"   Done: {done}")
    print(f"   Info keys: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
except Exception as e:
    print(f"\n❌ خطا: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
