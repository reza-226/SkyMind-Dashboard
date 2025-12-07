import sys
sys.path.append('./environments')
from uav_mec_env import UAVMECEnvironment
import numpy as np

# بدون پارامتر یا با پارامترهای پیش‌فرض
env = UAVMECEnvironment()

state = env.reset()
print(f"State shape: {state.shape}")
print(f"State type: {type(state)}")

# تست با action format مختلف
print("\n" + "="*50)
print("Testing Action Format:")

# فرمت 1: آرایه ساده 7 عنصری
action = np.array([0, 0.5, 0.3, 0.3, 0.4, 2.0, 1.5], dtype=np.float32)
print(f"\nAction shape: {action.shape}")
print(f"Action: {action}")

try:
    next_state, reward, done, info = env.step(action)
    print("✅ Action accepted!")
    print(f"Reward: {reward:.2f}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
