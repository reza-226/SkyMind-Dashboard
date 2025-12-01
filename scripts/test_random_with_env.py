"""
Test Random Policy with actual UAV-MEC environment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from algorithms.baselines.random_policy import RandomAgent

print("=" * 70)
print("🧪 Testing Random Policy with Environment")
print("=" * 70)

# Try to import environment
print("\n📦 Attempting to import environment...")
try:
    from environments.uav_mec_env import UAVMECEnv
    print("✅ environments.uav_mec_env.UAVMECEnv imported!")
    env_available = True
except ImportError as e:
    print(f"⚠️  Could not import UAVMECEnv: {e}")
    env_available = False

if not env_available:
    print("\n🔍 Trying alternative imports...")
    try:
        from core.env.environment import Environment
        print("✅ core.env.environment.Environment imported!")
        env_available = True
    except ImportError as e:
        print(f"❌ Could not import Environment: {e}")

if env_available:
    print("\n🎯 Environment found! Ready for baseline testing.")
    print("\n📝 Next steps:")
    print("   1. Initialize environment with proper config")
    print("   2. Run Random baseline for 10 test episodes")
    print("   3. Collect metrics and save results")
else:
    print("\n❌ No environment available for testing")
    print("📝 Manual setup needed - check environment configuration")

print("\n" + "=" * 70)
