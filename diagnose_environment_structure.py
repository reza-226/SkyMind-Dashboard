# diagnose_environment_structure.py
import sys
sys.path.append('environments')
from uav_mec_env import UAVMECEnvironment
import inspect

print("="*70)
print("🔍 Analyzing UAVMECEnvironment Structure")
print("="*70)

# بررسی signature
print("\n📋 Environment __init__ signature:")
sig = inspect.signature(UAVMECEnvironment.__init__)
print(f"   {sig}")

# ایجاد محیط (بدون level)
print("\n📋 Creating environment (default parameters)...")
env = UAVMECEnvironment()

# بررسی state
print("\n📋 Resetting environment...")
state = env.reset()

print(f"\n📐 State Properties:")
print(f"   Type: {type(state)}")
print(f"   Dimension: {len(state)}")
print(f"   Shape: {state.shape if hasattr(state, 'shape') else 'N/A'}")
print(f"   First 10 elements: {state[:10]}")
print(f"   Last 10 elements: {state[-10:]}")

# بررسی action space
print(f"\n🎮 Action Space:")
try:
    action_sample = env.action_space.sample()
    print(f"   Sample action type: {type(action_sample)}")
    print(f"   Sample action: {action_sample}")
    if hasattr(action_sample, '__len__'):
        print(f"   Action dimension: {len(action_sample)}")
except Exception as e:
    print(f"   Error sampling action: {e}")

# تست چند reset
print("\n🔄 Testing multiple resets (state consistency):")
for i in range(5):
    state_i = env.reset()
    print(f"   Reset {i+1}: dimension = {len(state_i)}")

# بررسی attributes محیط
print("\n🔍 Environment Attributes:")
attrs = [attr for attr in dir(env) if not attr.startswith('_')]
print(f"   Public attributes: {attrs[:20]}")  # اولین 20 تا

print("\n" + "="*70)
