#!/usr/bin/env python3
"""
Diagnostic script to check data files
"""

from pathlib import Path
import pickle
import json

# Check what files exist
results_dir = Path("results/realtime")
print("="*60)
print("Data Files Diagnostic")
print("="*60)

print(f"\n📁 Checking directory: {results_dir}")
print(f"   Directory exists: {results_dir.exists()}")

if results_dir.exists():
    print(f"\n📄 Files in directory:")
    for file in sorted(results_dir.iterdir()):
        size = file.stat().st_size / 1024  # KB
        print(f"   • {file.name} ({size:.1f} KB)")
else:
    print("   ❌ Directory does not exist!")

# Try to load cache
cache_file = results_dir / "realtime_cache.pkl"
print(f"\n🔍 Checking cache file: {cache_file.name}")
print(f"   Exists: {cache_file.exists()}")

if cache_file.exists():
    try:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
            print(f"   ✅ Successfully loaded!")
            print(f"   Keys: {list(cache.keys())}")
            if 'history' in cache:
                print(f"   History records: {len(cache['history'])}")
                if cache['history']:
                    first = cache['history'][0]
                    print(f"   First record keys: {list(first.keys())}")
    except Exception as e:
        print(f"   ❌ Error loading: {e}")

# Try to load pareto
pareto_file = results_dir / "pareto_snapshot.json"
print(f"\n🔍 Checking Pareto file: {pareto_file.name}")
print(f"   Exists: {pareto_file.exists()}")

if pareto_file.exists():
    try:
        with open(pareto_file, 'r') as f:
            pareto = json.load(f)
            print(f"   ✅ Successfully loaded!")
            print(f"   Solutions: {len(pareto)}")
            if pareto:
                first = pareto[0]
                print(f"   First solution keys: {list(first.keys())}")
    except Exception as e:
        print(f"   ❌ Error loading: {e}")

print("\n" + "="*60)
