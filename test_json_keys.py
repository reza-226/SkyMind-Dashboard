# test_json_keys.py
import json
from pathlib import Path

levels = ['level1_simple', 'level2_medium', 'level3_complex']

for level in levels:
    json_path = Path(f'models/{level}/training_history.json')
    if json_path.exists():
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"\n📁 {level}:")
        print(f"   کلیدها: {list(data.keys())}")
        # نمایش اولین مقدار هر کلید
        for key in list(data.keys())[:5]:
            val = data[key]
            if isinstance(val, list):
                print(f"   {key}: لیست با {len(val)} عنصر")
            else:
                print(f"   {key}: {type(val).__name__}")
    else:
        print(f"\n❌ {level}: فایل وجود ندارد")
