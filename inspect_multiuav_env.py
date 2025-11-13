"""
inspect_multiuav_env.py
بررسی ساختار MultiUAVEnv
"""

import sys
import inspect
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.env_multi import MultiUAVEnv

print("=" * 60)
print("MultiUAVEnv Inspection")
print("=" * 60)

# نمایش signature
sig = inspect.signature(MultiUAVEnv.__init__)
print(f"\n📋 __init__ signature:")
print(f"   {sig}")

print(f"\n📝 Parameters:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        default = param.default if param.default != inspect.Parameter.empty else 'No default'
        print(f"   - {param_name}: {default}")

# تست ساخت محیط
print(f"\n🧪 Testing environment creation...")
try:
    env = MultiUAVEnv()
    print("   ✅ Created with default parameters")
    
    # بررسی attributes
    print(f"\n🔍 Environment attributes:")
    for attr in ['n_uavs', 'num_uavs', 'n_agents', 'num_agents']:
        if hasattr(env, attr):
            value = getattr(env, attr)
            print(f"   ✅ {attr} = {value}")
    
    # بررسی state structure
    state = env.reset()
    print(f"\n📦 State structure:")
    if isinstance(state, dict):
        for key, value in state.items():
            if isinstance(value, (list, tuple)):
                print(f"   - {key}: length={len(value)}")
            else:
                print(f"   - {key}: {type(value).__name__}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
