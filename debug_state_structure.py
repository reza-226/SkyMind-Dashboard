"""
debug_state_structure.py
========================
بررسی دقیق ساختار State
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.env_multi import MultiUAVEnv
import numpy as np

def inspect_state_deeply():
    print("="*70)
    print("🔍 بررسی عمیق ساختار State")
    print("="*70)
    
    env = MultiUAVEnv(n_agents=3)
    
    # Reset و بررسی state اولیه
    print("\n📦 State بعد از reset():")
    state = env.reset()
    
    print(f"\nنوع state: {type(state)}")
    
    if isinstance(state, dict):
        print(f"\nکلیدهای موجود در state:")
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                print(f"  {key:20s}: shape={value.shape}, dtype={value.dtype}")
                print(f"                       sample={value.flatten()[:3]}...")
            else:
                print(f"  {key:20s}: {type(value).__name__} = {value}")
    
    # اجرای یک step
    print("\n\n🎬 اجرای یک step...")
    actions = []
    for i in range(3):
        v = 20.0
        theta = np.pi/4
        f = 2e9
        o = 0.7
        actions.append(np.array([v, theta, f, o], dtype=np.float32))
    
    step_result = env.step(actions)
    
    print(f"\nتعداد خروجی‌های step(): {len(step_result)}")
    
    if len(step_result) >= 4:
        next_state = step_result[0]
        rewards = step_result[1]
        dones = step_result[2]
        
        print(f"\n📦 Next State بعد از step():")
        if isinstance(next_state, dict):
            print(f"\nکلیدهای موجود:")
            for key, value in next_state.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key:20s}: shape={value.shape}")
                    print(f"                       min={np.min(value):.4f}, "
                          f"max={np.max(value):.4f}, "
                          f"mean={np.mean(value):.4f}")
                else:
                    print(f"  {key:20s}: {type(value).__name__} = {value}")
        
        print(f"\n💰 Rewards: {rewards}")
        print(f"🏁 Dones: {dones}")
    
    # بررسی attributes محیط
    print("\n\n🔧 Attributes قابل دسترس در محیط:")
    env_attrs = [attr for attr in dir(env) if not attr.startswith('_')]
    for attr in env_attrs[:20]:  # اولین 20 attribute
        try:
            value = getattr(env, attr)
            if not callable(value):
                print(f"  {attr:25s}: {type(value).__name__}")
        except:
            pass
    
    # بررسی info اگر وجود دارد
    if len(step_result) >= 5:
        info = step_result[4]
        print(f"\n📋 Info dictionary:")
        if isinstance(info, dict):
            for key, value in info.items():
                print(f"  {key:20s}: {value}")

if __name__ == "__main__":
    inspect_state_deeply()
