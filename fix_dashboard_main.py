# fix_dashboard_main.py
import os

dashboard_path = 'analysis/realtime/dashboard_complete.py'

print("="*80)
print("🔧 Fixing Dashboard __main__ Section...")
print("="*80)

with open(dashboard_path, 'r', encoding='utf-8') as f:
    content = f.read()

# پیدا کردن بخش مشکل‌دار
old_code = '''    if data_loader:
        print("✅ TrainingDataLoader connected successfully")
        try:
            summary = data_loader.get_summary_stats()
            print(f"📈 Total Episodes: {summary['total_episodes']}")
            print(f"🏆 Average Reward: {summary['avg_reward']:.2f}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load summary stats: {e}")'''

new_code = '''    if data_loader:
        print("✅ TrainingDataLoader connected successfully")
        try:
            # 🔥 لود کردن داده‌های level1
            print("📂 Loading level1 data...")
            data_loader.load_level_data('level1')
            
            summary = data_loader.get_summary_stats()
            print(f"📈 Total Episodes: {summary['total_episodes']}")
            print(f"🏆 Average Reward: {summary['avg_reward']:.2f}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load summary stats: {e}")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Fixed: Added load_level_data('level1') call")
else:
    print("⚠️ Pattern not found - manual fix needed")

# ذخیره
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Dashboard fixed successfully!")
print("="*80)
print("\n🚀 Now run:")
print("python analysis/realtime/dashboard_complete.py")
print("="*80)
