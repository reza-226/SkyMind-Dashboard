# debug_env_structure.py
import sys
sys.path.insert(0, 'D:/Payannameh/SkyMind-Dashboard')

from core.env_multi import MultiUAVEnv
import numpy as np
import inspect

env = MultiUAVEnv(n_agents=3)

print("=" * 60)
print("🔍 ساختار محیط MultiUAVEnv")
print("=" * 60)

# بررسی attribute‌ها
print("\n📋 Attributes موجود:")
attrs = [a for a in dir(env) if not a.startswith('_')]
for attr in attrs[:20]:  # 20 تای اول
    print(f"   - {attr}")

# بررسی متدها
print("\n🔧 Methods مهم:")
important_methods = ['reset', 'step', 'render', 'close']
for method in important_methods:
    if hasattr(env, method):
        sig = inspect.signature(getattr(env, method))
        print(f"   ✅ {method}{sig}")

# تست reset
print("\n🧪 تست reset():")
try:
    obs = env.reset()
    print(f"   ✅ موفق")
    print(f"   نوع: {type(obs)}")
    
    if isinstance(obs, dict):
        print(f"   کلیدها: {list(obs.keys())}")
        for k, v in list(obs.items())[:3]:
            print(f"      {k}: {type(v)} - shape={np.array(v).shape if hasattr(v, 'shape') or isinstance(v, (list, np.ndarray)) else 'N/A'}")
    elif isinstance(obs, (list, tuple)):
        print(f"   طول: {len(obs)}")
        for i in range(min(3, len(obs))):
            print(f"      [{i}]: {type(obs[i])} - {np.array(obs[i]).shape if isinstance(obs[i], (list, np.ndarray)) else obs[i]}")
    else:
        print(f"   شکل: {np.array(obs).shape if hasattr(obs, 'shape') else 'scalar'}")
        print(f"   نمونه: {obs}")
        
except Exception as e:
    print(f"   ❌ خطا: {e}")
    import traceback
    traceback.print_exc()

# بررسی action
print("\n🎮 بررسی Action:")
print(f"   n_agents: {env.n_agents if hasattr(env, 'n_agents') else 'N/A'}")

# تست action‌های مختلف
print("\n🧪 تست action formats:")
test_actions = [
    ("dict با tuple", {0: (0.5, 0.5), 1: (0.5, 0.5), 2: (0.5, 0.5)}),
    ("dict با array", {0: np.array([0.5, 0.5]), 1: np.array([0.5, 0.5]), 2: np.array([0.5, 0.5])}),
    ("list of tuple", [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]),
    ("list of array", [np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.5])]),
]

for name, action in test_actions:
    try:
        env.reset()
        next_obs, reward, done, info = env.step(action)
        print(f"   ✅ {name}: موفق")
        print(f"      reward type: {type(reward)}, done type: {type(done)}")
        break
    except Exception as e:
        print(f"   ❌ {name}: {str(e)[:50]}")

print("=" * 60)
