# scripts/inspect_environment.py
"""بررسی constructor محیط"""
import sys
from pathlib import Path
import inspect

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.env.environment import UAVMECEnvironment

print("="*70)
print("🔍 UAVMECEnvironment Constructor Inspection")
print("="*70)

# Get constructor signature
sig = inspect.signature(UAVMECEnvironment.__init__)
print(f"\n📋 Constructor signature:")
print(f"   {sig}\n")

print("📝 Parameters:")
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
    print(f"   • {param_name}: {default}")

# Try to create instance with no args
print("\n" + "="*70)
print("🧪 Testing instantiation...")
print("="*70)

try:
    env = UAVMECEnvironment()
    print("✅ Can create with no arguments")
    
    # Get state to find dimensions
    state = env.reset()
    print(f"\n📊 Environment info:")
    print(f"   • State type: {type(state)}")
    if isinstance(state, dict):
        print(f"   • State keys: {list(state.keys())}")
        print(f"   • State dim (first agent): {len(list(state.values())[0])}")
    else:
        print(f"   • State shape: {state.shape if hasattr(state, 'shape') else len(state)}")
    
except TypeError as e:
    print(f"❌ Cannot create with no args: {e}")
    print("\n💡 Trying with common parameters...")
    
    for params in [
        {'config': None},
        {'num_agents': 3},
        {'n_agents': 3},
    ]:
        try:
            env = UAVMECEnvironment(**params)
            print(f"   ✅ Works with: {params}")
            break
        except Exception as e:
            print(f"   ❌ Failed with {params}: {e}")
