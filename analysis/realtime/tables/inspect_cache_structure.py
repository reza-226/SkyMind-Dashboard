"""
inspect_cache_structure.py
=========================
Inspect the actual structure of realtime_cache.pkl
"""

import pickle
from pathlib import Path
import json

def inspect_cache():
    """Inspect cache structure"""
    project_root = Path(__file__).parent.parent.parent.parent
    cache_path = project_root / "analysis" / "realtime" / "realtime_cache.pkl"
    
    print("🔍 بررسی ساختار Cache...")
    print("=" * 60)
    
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    
    print(f"\n📦 نوع cache: {type(cache)}")
    
    if isinstance(cache, dict):
        print(f"\n🔑 کلیدهای موجود:")
        for key in cache.keys():
            value = cache[key]
            print(f"  - {key}: {type(value)}")
            if isinstance(value, (list, dict)):
                if isinstance(value, list) and len(value) > 0:
                    print(f"    → تعداد: {len(value)}")
                    print(f"    → نمونه اول: {type(value[0])}")
                elif isinstance(value, dict):
                    print(f"    → کلیدها: {list(value.keys())[:5]}")
    
    # Save structure to JSON for review
    structure = {
        "type": str(type(cache)),
        "keys": list(cache.keys()) if isinstance(cache, dict) else "Not a dict"
    }
    
    structure_path = project_root / "analysis" / "realtime" / "cache_structure.json"
    with open(structure_path, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 ساختار ذخیره شد در: {structure_path}")

if __name__ == "__main__":
    inspect_cache()
