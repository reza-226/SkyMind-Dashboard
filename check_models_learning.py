import json
import os

# مسیرهای فایل‌های JSON
levels = [
    'models/level1_simple/training_history.json',
    'models/level2_medium/training_history.json',
    'models/level3_complex/training_history.json'
]

project_root = r'D:\Payannameh\SkyMind-Dashboard'

print("🔍 بررسی محتوای فایل‌های JSON:\n")
print("="*80)

for level_path in levels:
    full_path = os.path.join(project_root, level_path)
    
    if not os.path.exists(full_path):
        print(f"❌ {level_path} - یافت نشد!\n")
        continue
    
    # خواندن فایل
    with open(full_path, 'r') as f:
        data = json.load(f)
    
    level_name = level_path.split('/')[1]  # مثلاً level1_simple
    
    print(f"\n📊 {level_name}")
    print(f"   📂 مسیر: {level_path}")
    print(f"   💾 حجم فایل: {os.path.getsize(full_path) / 1024:.2f} KB")
    
    # بررسی کلیدهای موجود
    print(f"   🔑 کلیدهای موجود: {list(data.keys())}")
    
    # بررسی تعداد داده‌ها
    if 'rewards' in data:
        print(f"   📈 تعداد Rewards: {len(data['rewards'])}")
        print(f"   🎯 محدوده Rewards: [{min(data['rewards']):.2f}, {max(data['rewards']):.2f}]")
        print(f"   📊 میانگین: {sum(data['rewards'])/len(data['rewards']):.2f}")
    
    if 'actor_losses' in data:
        print(f"   📉 Actor Loss: [{min(data['actor_losses']):.4f}, {max(data['actor_losses']):.4f}]")
    
    if 'critic_losses' in data:
        print(f"   📉 Critic Loss: [{min(data['critic_losses']):.4f}, {max(data['critic_losses']):.4f}]")
    
    print("   " + "-"*76)

print("\n" + "="*80)
print("✅ بررسی کامل شد!")
