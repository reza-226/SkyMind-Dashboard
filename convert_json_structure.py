# convert_json_structure.py
import json
import os

json_path = 'models/maddpg/training_history.json'
backup_path = 'models/maddpg/training_history_backup.json'

print("🔄 تبدیل ساختار JSON...")

# بک‌آپ گرفتن
with open(json_path, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

with open(backup_path, 'w', encoding='utf-8') as f:
    json.dump(original_data, f, indent=2, ensure_ascii=False)
print(f"✅ بک‌آپ ذخیره شد: {backup_path}")

# تبدیل ساختار
converted_data = {}

for episode_key, episode_data in original_data.items():
    converted_data[episode_key] = {
        'episode': episode_data.get('episode'),
        'reward': episode_data.get('avg_reward', 0),  # ✅ تبدیل
        'critic_loss': episode_data.get('critic_loss', 0),
        'noise': episode_data.get('noise_std', 0),    # ✅ تبدیل
        'buffer_size': episode_data.get('buffer_size', 0)
    }

# ذخیره ساختار جدید
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, indent=2, ensure_ascii=False)

print(f"✅ ساختار تبدیل شد!")
print(f"\n📊 نمونه داده جدید:")
print(f"   {converted_data['1']}")

# تایید
print("\n" + "=" * 60)
print("✅ Dashboard حالا می‌تونه داده‌ها رو بخونه!")
print("=" * 60)
