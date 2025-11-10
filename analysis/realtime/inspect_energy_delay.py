# ===============================================================
#  analysis/realtime/inspect_energy_delay.py
#  بررسی معیارهای انرژی و تاخیر
# ===============================================================

import pickle
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
CACHE_FILE = BASE_DIR / "realtime_cache.pkl"
PARETO_FILE = BASE_DIR / "pareto_snapshot.json"

print("=" * 70)
print("🔍 بررسی معیارهای انرژی و تاخیر")
print("=" * 70)

# ===============================================================
# 1. بررسی محتوای Cache
# ===============================================================

print("\n[1/3] بررسی realtime_cache.pkl...")

with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

print(f"\n📦 کلیدهای موجود در Cache:")
for key in cache.keys():
    print(f"  - {key}: {type(cache[key])}")

# ===============================================================
# 2. بررسی محتوای Pareto Solutions
# ===============================================================

print("\n[2/3] بررسی pareto_snapshot.json...")

with open(PARETO_FILE, 'r') as f:
    pareto_data = json.load(f)

print(f"\n📊 تعداد راه‌حل‌ها: {len(pareto_data['solutions'])}")
print(f"\n🔑 کلیدهای هر راه‌حل:")

if pareto_data['solutions']:
    first_solution = pareto_data['solutions'][0]
    for key, value in first_solution.items():
        print(f"  - {key}: {type(value).__name__} = {value}")

# ===============================================================
# 3. جستجوی معیارهای مرتبط با انرژی و تاخیر
# ===============================================================

print("\n[3/3] جستجوی معیارهای Energy و Delay...")

# بررسی در U, Δ, Ω
print("\n📈 تحلیل معیارهای موجود:")

solutions = pareto_data['solutions']
U_values = [s['U'] for s in solutions]
Delta_values = [s['Δ'] for s in solutions]
Omega_values = [s['Ω'] for s in solutions]

print(f"\n1️⃣ Utility (U):")
print(f"   - میانگین: {np.mean(U_values):.4f}")
print(f"   - بازه: [{min(U_values):.4f}, {max(U_values):.4f}]")
print(f"   ❓ آیا شامل انرژی است؟")

print(f"\n2️⃣ Error Rate (Δ):")
print(f"   - میانگین: {np.mean(Delta_values):.2f}%")
print(f"   - بازه: [{min(Delta_values):.2f}%, {max(Delta_values):.2f}%]")
print(f"   ❓ آیا مربوط به تاخیر است؟")

print(f"\n3️⃣ Stability (Ω):")
print(f"   - میانگین: {np.mean(Omega_values):.4f}")
print(f"   - بازه: [{min(Omega_values):.4f}, {max(Omega_values):.4f}]")
print(f"   ❓ آیا شامل انرژی/تاخیر است؟")

# ===============================================================
# 4. بررسی تعریف Utility Function
# ===============================================================

print("\n" + "=" * 70)
print("📚 بررسی تعریف تابع Utility در کد")
print("=" * 70)

# چک کردن فایل utility
utility_file = Path("analysis/pareto_convergence/dashboard.py")

if utility_file.exists():
    print(f"\n✅ فایل یافت شد: {utility_file}")
    print("\n🔎 جستجوی تعریف Utility...")
    
    with open(utility_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # جستجوی خطوط مربوط به utility
    lines = content.split('\n')
    utility_lines = []
    
    for i, line in enumerate(lines):
        if 'def compute_utility' in line.lower() or \
           'utility' in line.lower() and ('energy' in line.lower() or 'delay' in line.lower()):
            utility_lines.append((i+1, line))
    
    if utility_lines:
        print("\n📝 خطوط مرتبط یافت شده:")
        for line_num, line in utility_lines[:10]:  # اول 10 خط
            print(f"   Line {line_num}: {line.strip()}")
    else:
        print("\n⚠️  تعریف صریح Utility یافت نشد")

else:
    print(f"\n❌ فایل یافت نشد: {utility_file}")

# ===============================================================
# 5. بررسی معماری MATO_UAV
# ===============================================================

print("\n" + "=" * 70)
print("🏗️  بررسی معماری MATO_UAV")
print("=" * 70)

env_file = Path("env/mato_uav_v2.py")

if env_file.exists():
    print(f"\n✅ فایل محیط یافت شد: {env_file}")
    print("\n🔎 جستجوی محاسبات Energy و Delay...")
    
    with open(env_file, 'r', encoding='utf-8') as f:
        env_content = f.read()
    
    keywords = ['energy', 'delay', 'latency', 'consumption', 'power']
    found_keywords = {}
    
    for keyword in keywords:
        count = env_content.lower().count(keyword)
        if count > 0:
            found_keywords[keyword] = count
    
    if found_keywords:
        print("\n📊 کلمات کلیدی یافت شده:")
        for kw, count in found_keywords.items():
            print(f"   - '{kw}': {count} بار")
    else:
        print("\n⚠️  کلمات کلیدی مرتبط یافت نشد")

else:
    print(f"\n❌ فایل محیط یافت نشد: {env_file}")

# ===============================================================
# خلاصه و نتیجه‌گیری
# ===============================================================

print("\n" + "=" * 70)
print("📋 خلاصه یافته‌ها")
print("=" * 70)

print("""
🎯 وضعیت فعلی:
   - معیارهای ذخیره شده: U, Δ, Ω
   - معیارهای صریح Energy/Delay: نیاز به بررسی بیشتر

💡 سه حالت ممکن:

1️⃣  انرژی و تاخیر داخل U محاسبه شده‌اند
   ➜ نیاز به استخراج از تعریف Utility

2️⃣  در محیط محاسبه می‌شوند ولی ذخیره نشده‌اند
   ➜ نیاز به اصلاح کد ذخیره‌سازی

3️⃣  در شبیه‌سازی synthetic محاسبه نمی‌شوند
   ➜ نیاز به اضافه کردن محاسبات

🔧 قدم بعدی:
   - بررسی دقیق فرمول Utility
   - اضافه کردن log برای Energy/Delay
   - بازنگری کد شبیه‌سازی
""")

print("=" * 70)
