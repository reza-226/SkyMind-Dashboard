import pickle

# بارگذاری cache
with open('realtime_cache.pkl', 'rb') as f:
    cache_data = pickle.load(f)

# نمایش کلیدها
print("🔑 کلیدهای موجود در cache:")
print(cache_data.keys())
print("\n" + "="*70 + "\n")

# اگر 'pareto' وجود دارد، ستون‌هایش را نمایش بده
if 'pareto' in cache_data:
    df = cache_data['pareto']
    print("📊 ستون‌های موجود در DataFrame 'pareto':")
    print(df.columns.tolist())
    print(f"\n📏 تعداد رکوردها: {len(df)}")
    print(f"\n🔢 نمونه داده (5 ردیف اول):")
    print(df.head())
else:
    print("⚠️ کلید 'pareto' وجود ندارد!")
