# check_test_environment.py
import sys
sys.path.append('environments')
from uav_mec_env import UAVMECEnvironment

print("="*60)
print("🔍 Checking Test Environment Dimensions")
print("="*60)

# ساخت محیط دقیقاً مثل test_trained_model.py
env = UAVMECEnvironment()
state = env.reset()  # فقط state برمی‌گرداند

print(f"\n📐 State Type: {type(state)}")
print(f"   State Shape: {state.shape if hasattr(state, 'shape') else 'N/A'}")
print(f"   State Dimension: {len(state)}")
print(f"\n📊 First 10 values: {state[:10]}")

# چک کردن 5 reset
print("\n🔄 Checking 5 resets:")
for i in range(5):
    state = env.reset()
    print(f"   Reset {i+1}: dimension = {len(state)}")

env.close()

print("\n" + "="*60)
print("✅ Environment Check Complete")
print("="*60)
