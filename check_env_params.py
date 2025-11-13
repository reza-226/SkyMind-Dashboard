# check_env_params.py
import sys
sys.path.append('.')

from core.env_multi import MultiUAVEnv
import inspect

# بررسی signature
sig = inspect.signature(MultiUAVEnv.__init__)
print("MultiUAVEnv.__init__ parameters:")
print(sig)

# بررسی docstring
if MultiUAVEnv.__init__.__doc__:
    print("\nDocstring:")
    print(MultiUAVEnv.__init__.__doc__)

# تلاش برای ایجاد با پارامترهای پیش‌فرض
try:
    env = MultiUAVEnv()
    print(f"\n✅ محیط با موفقیت ایجاد شد")
    print(f"   - num_uavs: {env.num_uavs if hasattr(env, 'num_uavs') else 'N/A'}")
    print(f"   - state_dim: {env.observation_space.shape[0]}")
    print(f"   - action_dim: {env.action_space.shape[0]}")
except Exception as e:
    print(f"\n❌ خطا: {e}")
