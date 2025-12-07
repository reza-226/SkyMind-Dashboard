# fix_dashboard_initialization.py
import os

dashboard_path = 'analysis/realtime/dashboard_complete.py'

print("="*80)
print("🔧 Fixing Dashboard Initialization...")
print("="*80)

# خواندن فایل
with open(dashboard_path, 'r', encoding='utf-8') as f:
    content = f.read()

# پیدا کردن محل initialization
if 'data_loader = TrainingDataLoader()' in content:
    # اضافه کردن load_level_data بعد از initialization
    old_pattern = 'data_loader = TrainingDataLoader()'
    new_pattern = '''data_loader = TrainingDataLoader()
    # Load level1 data immediately
    data_loader.load_level_data('level1')
    print("📊 Level 1 data loaded at initialization")'''
    
    if 'load_level_data' not in content:
        content = content.replace(old_pattern, new_pattern)
        print("✅ Added load_level_data call")
    else:
        print("⚠️ load_level_data already exists")
else:
    print("❌ Could not find TrainingDataLoader initialization")

# ذخیره فایل
with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Dashboard fixed!")
print("="*80)
print("\n🚀 Now run: python analysis/realtime/dashboard_complete.py")
print("="*80)
