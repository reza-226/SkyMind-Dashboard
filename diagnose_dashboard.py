# diagnose_dashboard.py
import json
import os

json_path = 'models/maddpg/training_history.json'

print("=" * 60)
print("🔍 تشخیص مشکل Dashboard")
print("=" * 60)

# 1. بررسی وجود فایل
if not os.path.exists(json_path):
    print(f"❌ فایل وجود ندارد: {json_path}")
    print("   → باید training را اجرا کنی")
    exit()

print(f"✅ فایل وجود دارد: {json_path}")

# 2. بررسی سایز فایل
size = os.path.getsize(json_path)
print(f"📁 سایز فایل: {size:,} bytes")

if size < 100:
    print("⚠️ فایل خیلی کوچک است - احتمالاً خالی یا خراب")

# 3. خواندن محتوا
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ JSON معتبر است")
except json.JSONDecodeError as e:
    print(f"❌ JSON نامعتبر: {e}")
    exit()

# 4. بررسی ساختار
print(f"\n📊 ساختار داده:")
print(f"   - نوع: {type(data).__name__}")

if isinstance(data, dict):
    print(f"   - تعداد کلیدها: {len(data)}")
    
    # نمایش چند کلید اول
    keys = list(data.keys())[:5]
    print(f"   - کلیدهای اول: {keys}")
    
    # بررسی ساختار یک آیتم
    if keys:
        first_key = keys[0]
        first_val = data[first_key]
        print(f"\n   نمونه داده (کلید '{first_key}'):")
        print(f"   {first_val}")
        
elif isinstance(data, list):
    print(f"   - تعداد آیتم‌ها: {len(data)}")
    if data:
        print(f"   - نمونه اول: {data[0]}")

# 5. بررسی فیلدهای مورد انتظار Dashboard
print("\n" + "=" * 60)
print("🎯 بررسی فیلدهای مورد نیاز Dashboard:")
print("=" * 60)

expected_fields = ['reward', 'critic_loss', 'buffer_size', 'noise']

if isinstance(data, dict):
    # اگر dict با کلیدهای عددی
    sample = data.get('1') or data.get(1) or (list(data.values())[0] if data else {})
    
    if isinstance(sample, dict):
        for field in expected_fields:
            if field in sample:
                print(f"   ✅ {field}: موجود")
            else:
                print(f"   ❌ {field}: موجود نیست!")
    else:
        print(f"   ⚠️ ساختار غیرمنتظره: {type(sample)}")

print("\n" + "=" * 60)
