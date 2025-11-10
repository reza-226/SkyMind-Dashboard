# ===============================================================
#  analysis/realtime/diagnose_energy_delay.py
#  تشخیص منبع واقعی داده‌های Energy و Delay
# ===============================================================

import pickle
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
CACHE_FILE = BASE_DIR / "realtime_cache.pkl"
PARETO_FILE = BASE_DIR / "pareto_snapshot.json"

print("=" * 70)
print("🔬 تشخیص منبع داده‌های Energy و Delay")
print("=" * 70)

# بارگذاری داده‌ها
with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

with open(PARETO_FILE, 'r') as f:
    pareto = json.load(f)

solutions = pareto['solutions']

# استخراج داده‌ها
Energy_values = [s['Energy_J'] for s in solutions]
Delay_values = [s['Delay_ms'] for s in solutions]
Energy_Reduction = [s['Energy_Reduction_%'] for s in solutions]
Delay_Reduction = [s['Delay_Reduction_%'] for s in solutions]

print(f"\n📊 تعداد راه‌حل‌ها: {len(solutions)}")

# ===============================================================
# تست 1: آیا مقادیر ثابت هستند؟
# ===============================================================
print("\n" + "="*70)
print("تست 1️⃣: بررسی تنوع داده‌ها")
print("="*70)

unique_energy = len(set(Energy_values))
unique_delay = len(set(Delay_values))

print(f"\n🔹 Energy_J:")
print(f"   - تعداد مقادیر یکتا: {unique_energy}/{len(Energy_values)}")
print(f"   - میانگین: {np.mean(Energy_values):.4f} J")
print(f"   - انحراف معیار: {np.std(Energy_values):.4f}")
print(f"   - بازه: [{min(Energy_values):.4f}, {max(Energy_values):.4f}]")

if unique_energy == 1:
    print("   ⚠️  همه مقادیر یکسان هستند → احتمالاً placeholder")
elif unique_energy < 10:
    print("   ⚠️  تنوع بسیار کم → احتمالاً synthetic")
else:
    print("   ✅ تنوع قابل قبول → احتمالاً واقعی")

print(f"\n🔹 Delay_ms:")
print(f"   - تعداد مقادیر یکتا: {unique_delay}/{len(Delay_values)}")
print(f"   - میانگین: {np.mean(Delay_values):.2f} ms")
print(f"   - انحراف معیار: {np.std(Delay_values):.2f}")
print(f"   - بازه: [{min(Delay_values):.2f}, {max(Delay_values):.2f}]")

if unique_delay == 1:
    print("   ⚠️  همه مقادیر یکسان هستند → احتمالاً placeholder")
elif unique_delay < 10:
    print("   ⚠️  تنوع بسیار کم → احتمالاً synthetic")
else:
    print("   ✅ تنوع قابل قبول → احتمالاً واقعی")

# ===============================================================
# تست 2: آیا با U, Δ, Ω همبستگی دارند؟
# ===============================================================
print("\n" + "="*70)
print("تست 2️⃣: بررسی همبستگی با معیارهای اصلی")
print("="*70)

U_values = [s['U'] for s in solutions]
Delta_values = [s['Δ'] for s in solutions]
Omega_values = [s['Ω'] for s in solutions]

# محاسبه همبستگی
corr_E_U = np.corrcoef(Energy_values, U_values)[0, 1]
corr_E_Delta = np.corrcoef(Energy_values, Delta_values)[0, 1]
corr_E_Omega = np.corrcoef(Energy_values, Omega_values)[0, 1]

corr_D_U = np.corrcoef(Delay_values, U_values)[0, 1]
corr_D_Delta = np.corrcoef(Delay_values, Delta_values)[0, 1]
corr_D_Omega = np.corrcoef(Delay_values, Omega_values)[0, 1]

print(f"\n🔹 همبستگی Energy با:")
print(f"   - Utility (U): {corr_E_U:+.4f}")
print(f"   - Error Rate (Δ): {corr_E_Delta:+.4f}")
print(f"   - Stability (Ω): {corr_E_Omega:+.4f}")

print(f"\n🔹 همبستگی Delay با:")
print(f"   - Utility (U): {corr_D_U:+.4f}")
print(f"   - Error Rate (Δ): {corr_D_Delta:+.4f}")
print(f"   - Stability (Ω): {corr_D_Omega:+.4f}")

# تفسیر
print("\n💡 تفسیر:")
if abs(corr_E_U) > 0.7:
    print("   - Energy به شدت با Utility همبسته است")
if abs(corr_D_Delta) > 0.7:
    print("   - Delay به شدت با Error Rate همبسته است")

if abs(corr_E_U) < 0.1 and abs(corr_D_Delta) < 0.1:
    print("   ⚠️  هیچ همبستگی معنی‌داری یافت نشد → احتمالاً random/placeholder")

# ===============================================================
# تست 3: بررسی الگوی زمانی
# ===============================================================
print("\n" + "="*70)
print("تست 3️⃣: بررسی روند زمانی")
print("="*70)

# بررسی 100 اپیزود اول و آخر
first_100_E = Energy_values[:100]
last_100_E = Energy_values[-100:]
first_100_D = Delay_values[:100]
last_100_D = Delay_values[-100:]

improvement_E = ((np.mean(first_100_E) - np.mean(last_100_E)) / np.mean(first_100_E)) * 100
improvement_D = ((np.mean(first_100_D) - np.mean(last_100_D)) / np.mean(first_100_D)) * 100

print(f"\n🔹 Energy:")
print(f"   - اول 100 اپیزود: {np.mean(first_100_E):.4f} J")
print(f"   - آخر 100 اپیزود: {np.mean(last_100_E):.4f} J")
print(f"   - بهبود: {improvement_E:+.2f}%")

print(f"\n🔹 Delay:")
print(f"   - اول 100 اپیزود: {np.mean(first_100_D):.2f} ms")
print(f"   - آخر 100 اپیزود: {np.mean(last_100_D):.2f} ms")
print(f"   - بهبود: {improvement_D:+.2f}%")

print("\n💡 تفسیر:")
if abs(improvement_E) < 1 and abs(improvement_D) < 1:
    print("   ⚠️  هیچ روند بهبودی مشاهده نشد → احتمالاً placeholder")
elif improvement_E > 5 and improvement_D > 5:
    print("   ✅ روند بهبود واضح → داده‌ها واقعی به نظر می‌رسند")

# ===============================================================
# تست 4: بررسی فیلدهای Reduction
# ===============================================================
print("\n" + "="*70)
print("تست 4️⃣: بررسی فیلدهای Reduction")
print("="*70)

unique_E_reduction = len(set(Energy_Reduction))
unique_D_reduction = len(set(Delay_Reduction))

print(f"\n🔹 Energy_Reduction_%:")
print(f"   - تعداد مقادیر یکتا: {unique_E_reduction}")
print(f"   - بازه: [{min(Energy_Reduction):.2f}%, {max(Energy_Reduction):.2f}%]")

print(f"\n🔹 Delay_Reduction_%:")
print(f"   - تعداد مقادیر یکتا: {unique_D_reduction}")
print(f"   - بازه: [{min(Delay_Reduction):.2f}%, {max(Delay_Reduction):.2f}%]")

if unique_E_reduction == 1 and Energy_Reduction[0] == 0.0:
    print("   ⚠️  همه مقادیر Reduction صفر هستند → احتمالاً محاسبه نشده")

# ===============================================================
# نتیجه‌گیری نهایی
# ===============================================================
print("\n" + "="*70)
print("🎯 نتیجه‌گیری نهایی")
print("="*70)

score = 0

# امتیازدهی
if unique_energy > 100:
    score += 1
if unique_delay > 100:
    score += 1
if abs(improvement_E) > 5:
    score += 1
if abs(improvement_D) > 5:
    score += 1
if abs(corr_E_U) > 0.3 or abs(corr_D_Delta) > 0.3:
    score += 1

print(f"\n📊 امتیاز کیفیت داده: {score}/5")

if score >= 4:
    print("\n✅ داده‌های Energy و Delay به نظر واقعی و معتبر هستند")
    print("   → می‌توانید از اسکریپت‌های گزارش‌ساز استفاده کنید")
elif score >= 2:
    print("\n⚠️  داده‌ها احتمالاً synthetic یا محاسبه‌شده از فرمول‌های ساده هستند")
    print("   → نمودارها تولید می‌شوند اما باید با احتیاط تفسیر شوند")
else:
    print("\n❌ داده‌ها احتمالاً placeholder هستند")
    print("   → نیاز به بازبینی کد تولید داده")

print("\n" + "="*70)
print("✨ تشخیص کامل شد")
print("="*70)
