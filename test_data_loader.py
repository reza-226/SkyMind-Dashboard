# test_data_loader.py
import sys
import os

# اضافه کردن مسیر پروژه
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from dashboard.data_loader import TrainingDataLoader

print("="*80)
print("🧪 Testing TrainingDataLoader...")
print("="*80)

loader = TrainingDataLoader()
print(f"✅ Loader initialized")

print(f"\n📂 Loading level1 data...")
loader.load_level_data('level1')

stats = loader.get_summary_stats()
print(f"\n📊 Summary Stats:")
print(f"   Total Episodes: {stats['total_episodes']}")
print(f"   Average Reward: {stats['avg_reward']:.2f}")
print(f"   Success Rate: {stats['success_rate']:.1f}%")

print("="*80)
