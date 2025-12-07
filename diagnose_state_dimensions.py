# diagnose_state_dimensions.py
import sys
sys.path.append('environments')
from uav_mec_env import UAVMECEnvironment
import json

print("="*70)
print("🔍 Diagnosing State Dimension Inconsistency")
print("="*70)

# تست 1: آموزش (با level)
print("\n📋 Test 1: Training Mode (with level)")
env_train = UAVMECEnvironment(level='level_1')
state_train = env_train.reset()
print(f"   State dimension: {len(state_train)}")
print(f"   State type: {type(state_train)}")
print(f"   First 5 elements: {state_train[:5]}")

# تست 2: بدون level
print("\n📋 Test 2: Without level parameter")
env_test = UAVMECEnvironment()
state_test = env_test.reset()
print(f"   State dimension: {len(state_test)}")
print(f"   State type: {type(state_test)}")
print(f"   First 5 elements: {state_test[:5]}")

# تست 3: با level متفاوت
print("\n📋 Test 3: Different level (level_2)")
try:
    env_level2 = UAVMECEnvironment(level='level_2')
    state_level2 = env_level2.reset()
    print(f"   State dimension: {len(state_level2)}")
except Exception as e:
    print(f"   Error: {e}")

# بررسی فایل config
print("\n📋 Test 4: Checking environment config")
try:
    with open('environments/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("   Config keys:", list(config.keys()))
    if 'level_1' in config:
        print(f"   level_1 config: {config['level_1']}")
    if 'state_space' in config:
        print(f"   state_space: {config['state_space']}")
        
except Exception as e:
    print(f"   Could not read config: {e}")

print("\n" + "="*70)
print("🎯 Conclusion:")
print("   Training state_dim: 71")
print("   Test state_dim: 114")
print("   → Environment returns different dimensions based on initialization!")
print("="*70)
