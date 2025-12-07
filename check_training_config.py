# check_training_config.py
import os

print("="*70)
print("🔍 Checking Training Script Configuration")
print("="*70)

# بررسی فایل آموزش
train_file = "train_4layer_3level.py"

if os.path.exists(train_file):
    print(f"\n📄 Reading {train_file}...")
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # جستجوی state_dim
    lines = content.split('\n')
    print("\n🔍 Lines containing 'state_dim':")
    for i, line in enumerate(lines, 1):
        if 'state_dim' in line.lower():
            print(f"   Line {i}: {line.strip()}")
    
    # جستجوی hard-coded values
    print("\n🔍 Lines containing '71':")
    for i, line in enumerate(lines, 1):
        if '71' in line and not line.strip().startswith('#'):
            print(f"   Line {i}: {line.strip()}")
    
    print("\n🔍 Lines containing '537':")
    for i, line in enumerate(lines, 1):
        if '537' in line and not line.strip().startswith('#'):
            print(f"   Line {i}: {line.strip()}")

else:
    print(f"\n❌ {train_file} not found!")

print("\n" + "="*70)
