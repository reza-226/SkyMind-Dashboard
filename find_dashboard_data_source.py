# find_dashboard_data_source.py
import os
import re

dashboard_file = 'dashboard_complete.py'

print("=" * 60)
print("🔍 جستجوی مسیر داده در Dashboard")
print("=" * 60)

with open(dashboard_file, 'r', encoding='utf-8') as f:
    content = f.read()

# پیدا کردن تمام مسیرهای JSON
json_patterns = [
    r'["\']([^"\']*\.json)["\']',
    r'training_history',
    r'models/maddpg',
    r'load.*json',
]

print("\n📁 مسیرهای پیدا شده:")
for pattern in json_patterns:
    matches = re.findall(pattern, content, re.IGNORECASE)
    if matches:
        for m in set(matches):
            print(f"   → {m}")

# پیدا کردن توابع load
print("\n📖 توابع خواندن داده:")
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'json' in line.lower() and ('load' in line.lower() or 'open' in line.lower() or 'read' in line.lower()):
        print(f"   خط {i+1}: {line.strip()[:80]}")

print("\n" + "=" * 60)
