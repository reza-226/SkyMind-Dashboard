#!/usr/bin/env python3
"""اسکریپت تشخیص نام کلاس محیط"""

import re
from pathlib import Path

# مسیر فایل
env_file = Path("../core/env_multi.py")

if not env_file.exists():
    print("❌ فایل env_multi.py پیدا نشد!")
    exit(1)

# خواندن فایل
with open(env_file, 'r', encoding='utf-8') as f:
    content = f.read()

# جستجوی کلاس‌ها
pattern = r'^class\s+(\w+)\s*\('
matches = re.findall(pattern, content, re.MULTILINE)

if matches:
    print("✅ کلاس‌های پیدا شده:")
    for i, class_name in enumerate(matches, 1):
        print(f"   {i}. {class_name}")
    
    print(f"\n🎯 احتمالاً نام کلاس اصلی: {matches[0]}")
else:
    print("❌ هیچ کلاسی پیدا نشد!")
    print("\n📝 خطوط ابتدایی فایل:")
    lines = content.split('\n')[:30]
    for i, line in enumerate(lines, 1):
        if 'class' in line.lower():
            print(f"   {i}: {line}")
