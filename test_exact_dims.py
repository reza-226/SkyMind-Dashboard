# test_final_dimensions.py
import sys
import os
import numpy as np

# اضافه کردن مسیر به sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from environments.uav_mec_env import UAVMECEnvironment

print("="*60)
print("🔬 TEST: Environment Dimensions")
print("="*60)

# ساخت محیط با پارامترهای پیش‌فرض
try:
    env = UAVMECEnvironment()
    print("✅ Environment created successfully!")
except Exception as e:
    print(f"❌ Error creating environment: {e}")
    print("\n📋 Trying with minimal parameters...")
    try:
        env = UAVMECEnvironment(num_uavs=1)
        print("✅ Environment created with num_uavs=1")
    except Exception as e2:
        print(f"❌ Still failed: {e2}")
        sys.exit(1)

# تست Reset
print("\n" + "="*60)
print("📊 Testing env.reset()")
print("="*60)

try:
    result = env.reset()
    print(f"✅ Reset successful!")
    print(f"📦 Type: {type(result)}")
    
    if isinstance(result, tuple):
        state = result[0]
        print(f"📦 Tuple format (state, info)")
        print(f"   State type: {type(state)}")
        print(f"   State shape: {state.shape if hasattr(state, 'shape') else len(state)}")
    else:
        state = result
        print(f"📦 Direct state format")
        print(f"   State type: {type(state)}")
        print(f"   State shape: {state.shape if hasattr(state, 'shape') else len(state)}")
    
    print(f"\n🎯 STATE DIMENSION = {state.shape[0] if hasattr(state, 'shape') else len(state)}")
    
except Exception as e:
    print(f"❌ Reset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# تست Step با Action
print("\n" + "="*60)
print("🎮 Testing env.step()")
print("="*60)

# تلاش با action های مختلف
action_tests = [
    ("Array shape (7,)", np.random.randn(7)),
    ("Array shape (11,)", np.random.randn(11)),
    ("Dict format", {
        'offload': 0,
        'cpu': 0.5,
        'bandwidth': np.array([0.33, 0.33, 0.34]),
        'move': np.array([1.0, 1.0])
    }),
]

for test_name, action in action_tests:
    print(f"\n🧪 Test: {test_name}")
    try:
        result = env.step(action)
        print(f"   ✅ Success!")
        
        if isinstance(result, tuple) and len(result) >= 2:
            next_state = result[0]
            print(f"   📦 Next state shape: {next_state.shape if hasattr(next_state, 'shape') else len(next_state)}")
            print(f"   🎯 ACTION DIMENSION = {len(action) if isinstance(action, np.ndarray) else 'dict'}")
            break
        else:
            print(f"   ⚠️  Unexpected result format: {type(result)}")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")

print("\n" + "="*60)
print("✅ TEST COMPLETE")
print("="*60)
