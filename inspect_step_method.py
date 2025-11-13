# inspect_step_method.py
import sys
sys.path.insert(0, 'D:/Payannameh/SkyMind-Dashboard')

from core.env_multi import MultiUAVEnv
import inspect

env = MultiUAVEnv(n_agents=3)

print("=" * 60)
print("🔍 بررسی متد step()")
print("=" * 60)

# نمایش سیگنچر
sig = inspect.signature(env.step)
print(f"\n📋 Signature: step{sig}")

# نمایش کد
source = inspect.getsource(env.step)
print(f"\n💻 کد متد step:\n")
print(source[:1500])  # 1500 کاراکتر اول

print("=" * 60)
