# نام فایل: find_training_files.py
import os
from pathlib import Path

print("🔍 Searching for training files...\n")
print("="*70)

# مسیرهای احتمالی
search_paths = [
    Path("."),
    Path("models"),
    Path("results"),
    Path("outputs"),
    Path("checkpoints"),
]

# فایل‌های مورد نظر
target_files = [
    "training_history.json",
    "*.pt",
    "*.pth",
    "*.pkl",
    "checkpoint*",
]

found_files = {}

for search_path in search_paths:
    if not search_path.exists():
        continue
    
    print(f"\n📂 Searching in: {search_path}")
    
    # جستجو در مسیر و زیرمسیرها
    for root, dirs, files in os.walk(search_path):
        root_path = Path(root)
        
        for file in files:
            file_path = root_path / file
            
            # چک کردن نوع فایل
            if any([
                file.endswith('.json') and 'training' in file.lower(),
                file.endswith('.pt'),
                file.endswith('.pth'),
                file.endswith('.pkl') and 'training' in file.lower(),
                file.startswith('checkpoint'),
                'model' in file.lower() and (file.endswith('.pt') or file.endswith('.pth')),
            ]):
                size_mb = file_path.stat().st_size / (1024 * 1024)
                relative_path = file_path.relative_to(Path("."))
                
                if str(relative_path) not in found_files:
                    found_files[str(relative_path)] = {
                        'path': file_path,
                        'size': size_mb,
                        'type': file_path.suffix
                    }
                    print(f"   ✅ Found: {relative_path} ({size_mb:.2f} MB)")

print("\n" + "="*70)
print(f"\n📊 Summary: Found {len(found_files)} training-related files\n")

if not found_files:
    print("❌ No training files found!")
    print("\n💡 Possible reasons:")
    print("   1. Training was not completed successfully")
    print("   2. Files are in a different location")
    print("   3. train_sequential_levels.py did not run properly")
    print("\n🔧 Next steps:")
    print("   1. Check if train_sequential_levels.py exists")
    print("   2. Run: python train_sequential_levels.py")
    print("   3. Wait for training to complete")
else:
    print("📁 Files found at:")
    for path, info in found_files.items():
        print(f"   • {path}")
    
    # بررسی training_history.json
    training_json = None
    for path, info in found_files.items():
        if 'training_history.json' in path:
            training_json = info['path']
            break
    
    if training_json:
        print(f"\n✅ training_history.json found at: {training_json}")
        print("   You can now run: python view_training_results.py")
    else:
        print("\n⚠️ training_history.json NOT found!")
        print("   Checking for model checkpoints...")
        
        model_files = [p for p in found_files.keys() if '.pt' in p or '.pth' in p]
        if model_files:
            print(f"   ✅ Found {len(model_files)} model checkpoint(s)")
            print("   💡 Training may have completed but history file missing")

print("\n" + "="*70)
