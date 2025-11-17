# نام فایل: check_cache_structure.py
import pickle

# خواندن کش
with open('analysis/realtime/realtime_cache.pkl', 'rb') as f:
    data = pickle.load(f)

print("📦 Cache Structure:")
print("=" * 60)

# نمایش کلیدها
if isinstance(data, dict):
    print("\n🔑 Available Keys:")
    for key in data.keys():
        print(f"  • {key}")
    
    print("\n📊 Data Info:")
    for key, value in data.items():
        if hasattr(value, '__len__'):
            print(f"  • {key}: type={type(value).__name__}, length={len(value)}")
        else:
            print(f"  • {key}: type={type(value).__name__}, value={value}")
    
    # نمایش نمونه داده
    print("\n📌 Sample Data:")
    for key, value in data.items():
        if hasattr(value, '__len__') and len(value) > 0:
            print(f"\n  {key}:")
            if isinstance(value, (list, tuple)):
                print(f"    First 3 items: {value[:3]}")
            elif isinstance(value, dict):
                for k, v in list(value.items())[:3]:
                    print(f"    {k}: {v}")
else:
    print(f"⚠️ Data type: {type(data)}")
    print(f"Data content:\n{data}")
