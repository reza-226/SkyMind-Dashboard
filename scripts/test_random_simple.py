"""
Simple test for Random Policy without full environment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.baselines.random_policy import RandomAgent
import numpy as np

print("=" * 60)
print("🧪 Testing RandomAgent...")
print("=" * 60)

# Test agent initialization
agent = RandomAgent(offload_dim=5, continuous_dim=6)
print("\n✅ Agent created successfully")
print(f"   - Offload dimensions: {agent.offload_dim}")
print(f"   - Continuous dimensions: {agent.continuous_dim}")

# Test action selection
print("\n" + "=" * 60)
print("📊 Testing action selection (5 samples):")
print("=" * 60)

for i in range(5):
    action = agent.select_action(state=None)
    
    print(f"\n🎲 Sample {i+1}:")
    print(f"   Offload: {action['offload']} (should be 0-4)")
    print(f"   CPU: {action['cpu']:.4f} (should be 0-1)")
    print(f"   Bandwidth: [{action['bandwidth'][0]:.3f}, {action['bandwidth'][1]:.3f}, {action['bandwidth'][2]:.3f}]")
    print(f"   Bandwidth sum: {action['bandwidth'].sum():.6f} (should be ≈1.0)")
    print(f"   Move: [{action['move'][0]:+.3f}, {action['move'][1]:+.3f}] (should be -5 to +5)")
    
    # Validation
    assert 0 <= action['offload'] < 5, "Offload out of range!"
    assert 0 <= action['cpu'] <= 1, "CPU out of range!"
    assert abs(action['bandwidth'].sum() - 1.0) < 1e-6, "Bandwidth not normalized!"
    assert all(-5 <= m <= 5 for m in action['move']), "Movement out of range!"

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
