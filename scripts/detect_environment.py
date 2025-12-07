"""
Auto-detect available environments in the project
"""

import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("🔍 Auto-detecting Environments")
print("=" * 70)

# Possible environment locations
env_modules = [
    ("environments.uav_mec_env", ["UAVMECEnv", "UAVMECEnvironment", "Environment", "Env"]),
    ("core.env.environment", ["Environment", "BaseEnvironment", "MADDPGEnv"]),
    ("core.env_multi", ["MultiAgentEnv", "Environment"]),
    ("pettingzoo_env_maddpg", ["MADDPGEnv", "PettingZooEnv"]),
]

found_envs = []

for module_name, class_names in env_modules:
    print(f"\n📦 Checking module: {module_name}")
    try:
        module = importlib.import_module(module_name)
        print(f"   ✅ Module imported successfully")
        
        # List all classes in module
        module_classes = [name for name in dir(module) if not name.startswith('_')]
        print(f"   📋 Available items: {module_classes[:10]}...")
        
        # Try to find environment classes
        for class_name in class_names:
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                print(f"   ✅ Found class: {class_name}")
                found_envs.append((module_name, class_name, cls))
                
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("📊 Summary of Found Environments:")
print("=" * 70)

if found_envs:
    for i, (module, cls_name, cls) in enumerate(found_envs, 1):
        print(f"\n{i}. {module}.{cls_name}")
        print(f"   Type: {type(cls)}")
        
        # Try to inspect the class
        if hasattr(cls, '__init__'):
            import inspect
            try:
                sig = inspect.signature(cls.__init__)
                print(f"   Init signature: {sig}")
            except:
                print(f"   Init: (signature unavailable)")
else:
    print("\n❌ No environments found!")
    print("\n💡 Suggestions:")
    print("   1. Check if environment files exist")
    print("   2. Verify class names in source files")
    print("   3. Check for syntax errors in environment modules")

print("\n" + "=" * 70)
