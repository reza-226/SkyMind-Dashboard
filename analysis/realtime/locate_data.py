#!/usr/bin/env python3
"""
Locate and verify data files for report generation
"""

from pathlib import Path
import pickle
import json
import os

print("="*60)
print("🔍 Data Files Location Analysis")
print("="*60)

# Check all possible locations
locations = [
    "results/realtime",
    "results",
    "experiments/realtime",
    "experiments/realtime/results",
    "data"
]

print("\n📁 Checking directories:")
for loc in locations:
    p = Path(loc)
    exists = "✅" if p.exists() else "❌"
    print(f"   {exists} {loc}")
    if p.exists() and p.is_dir():
        files = list(p.iterdir())
        if files:
            print(f"      Contents ({len(files)} items):")
            for f in files[:5]:
                print(f"         • {f.name}")

# Search for specific files
print("\n🔍 Searching for data files:")
target_files = [
    ("realtime_cache.pkl", "Cache file"),
    ("pareto_snapshot.json", "Pareto front"),
    ("training_metrics.npz", "Training metrics")
]

for filename, description in target_files:
    found = list(Path('.').rglob(filename))
    print(f"\n   {description} ({filename}):")
    if found:
        for f in found:
            size = f.stat().st_size / 1024
            print(f"      ✅ {f} ({size:.1f} KB)")
    else:
        print(f"      ❌ Not found")

# Check if we need to run training first
print("\n" + "="*60)
print("📊 Analysis:")

has_pkl = list(Path('.').rglob('realtime_cache.pkl'))
has_json = list(Path('.').rglob('pareto_snapshot.json'))

if not has_pkl and not has_json:
    print("⚠️  No data files found!")
    print("\n💡 Solution:")
    print("   You need to run the training/simulation first:")
    print("   1. Run: python -m experiments.realtime.run_mato_v2")
    print("   2. Wait for it to generate results")
    print("   3. Then run report generators")
else:
    print("✅ Data files found!")
    if has_pkl:
        print(f"   Cache: {has_pkl[0]}")
    if has_json:
        print(f"   Pareto: {has_json[0]}")

print("="*60)
