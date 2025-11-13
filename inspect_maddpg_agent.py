"""
inspect_maddpg_agent.py
بررسی ساختار MADDPG_Agent
"""

import sys
import inspect
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from agents.agent_maddpg_multi import MADDPG_Agent

print("=" * 60)
print("MADDPG_Agent Inspection")
print("=" * 60)

# نمایش signature
sig = inspect.signature(MADDPG_Agent.__init__)
print(f"\n📋 __init__ signature:")
print(f"   {sig}")

print(f"\n📝 Parameters:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        default = param.default if param.default != inspect.Parameter.empty else 'No default'
        print(f"   - {param_name}: {default}")

# نمایش docstring
if MADDPG_Agent.__init__.__doc__:
    print(f"\n📖 Docstring:")
    print("   " + "\n   ".join(MADDPG_Agent.__init__.__doc__.strip().split('\n')))

# بررسی کلاس
print(f"\n🔍 Class attributes and methods:")
for attr in dir(MADDPG_Agent):
    if not attr.startswith('_'):
        print(f"   - {attr}")

print("\n" + "=" * 60)
