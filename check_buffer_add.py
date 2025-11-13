# check_buffer_add.py
"""
بررسی متد add در ReplayBuffer
"""

import inspect
from agents.agent_maddpg_multi import ReplayBuffer

print("="*70)
print("🔍 بررسی ReplayBuffer.add")
print("="*70)

# دریافت source code
source = inspect.getsource(ReplayBuffer.add)
print(source)

print("\n" + "="*70)
print("🔍 بررسی ReplayBuffer.sample")
print("="*70)

source = inspect.getsource(ReplayBuffer.sample)
print(source)
