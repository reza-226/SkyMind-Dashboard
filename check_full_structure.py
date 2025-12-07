# check_full_structure.py
import os
from pathlib import Path

BASE_DIR = Path(r"D:\Payannameh\SkyMind-Dashboard\content\drive\MyDrive\uav_mec\results")

print("=" * 60)
print("🔍 Full Directory Structure")
print("=" * 60)

if not BASE_DIR.exists():
    print(f"❌ Base directory not found: {BASE_DIR}")
else:
    print(f"✅ Base directory exists: {BASE_DIR}\n")
    
    for level in range(1, 5):
        level_dir = BASE_DIR / f"level{level}"
        
        print(f"\n{'='*60}")
        print(f"📁 Level {level}")
        print(f"{'='*60}")
        
        if not level_dir.exists():
            print(f"   ❌ Directory not found")
            continue
        
        # لیست تمام runs
        runs = sorted([d for d in level_dir.iterdir() if d.is_dir()])
        
        if not runs:
            print(f"   ❌ No run directories found")
            continue
        
        for run_dir in runs:
            print(f"\n   📂 Run: {run_dir.name}")
            
            # بررسی محتویات
            contents = list(run_dir.iterdir())
            
            if not contents:
                print(f"      ⚠️  Empty directory")
                continue
            
            for item in sorted(contents):
                if item.is_dir():
                    # اگر پوشه checkpoints باشد
                    if item.name == "checkpoints":
                        checkpoints = list(item.glob("checkpoint_*.pt"))
                        if checkpoints:
                            checkpoints.sort()
                            print(f"      ✅ {item.name}/ ({len(checkpoints)} files)")
                            print(f"         Latest: {checkpoints[-1].name}")
                        else:
                            print(f"      ⚠️  {item.name}/ (empty)")
                    else:
                        print(f"      📁 {item.name}/")
                else:
                    # فایل‌های دیگر
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"      📄 {item.name} ({size_mb:.2f} MB)")

print("\n" + "=" * 60)
