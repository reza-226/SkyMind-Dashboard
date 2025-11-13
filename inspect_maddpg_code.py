"""
inspect_maddpg_code.py
بررسی کد agent_maddpg_multi.py برای فهمیدن ساختار act()
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# خواندن کد
code_path = Path("agents/agent_maddpg_multi.py")

print("=" * 70)
print("📄 Reading agent_maddpg_multi.py")
print("=" * 70)

if code_path.exists():
    with open(code_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # پیدا کردن متد act()
    lines = code.split('\n')
    
    in_act_method = False
    act_method_lines = []
    indent_level = 0
    
    for i, line in enumerate(lines, 1):
        # شروع متد act
        if 'def act(' in line:
            in_act_method = True
            indent_level = len(line) - len(line.lstrip())
            act_method_lines.append((i, line))
        elif in_act_method:
            current_indent = len(line) - len(line.lstrip())
            
            # اگر خط خالی است یا indentation بیشتر/مساوی است
            if line.strip() == '' or current_indent > indent_level:
                act_method_lines.append((i, line))
            else:
                # تمام شد
                break
    
    if act_method_lines:
        print("\n🔍 Found act() method:\n")
        for line_num, line in act_method_lines:
            print(f"{line_num:4d} | {line}")
    else:
        print("\n⚠️  Could not find act() method")
        print("\n📋 Showing first 100 lines of code:\n")
        for i, line in enumerate(lines[:100], 1):
            print(f"{i:4d} | {line}")
    
    # بررسی __init__
    print("\n" + "=" * 70)
    print("🔍 Looking for __init__ method:")
    print("=" * 70)
    
    in_init = False
    init_lines = []
    indent_level = 0
    
    for i, line in enumerate(lines, 1):
        if 'def __init__(' in line:
            in_init = True
            indent_level = len(line) - len(line.lstrip())
            init_lines.append((i, line))
        elif in_init:
            current_indent = len(line) - len(line.lstrip())
            
            if line.strip() == '' or current_indent > indent_level:
                init_lines.append((i, line))
                
                # فقط 30 خط اول __init__ رو نشون بده
                if len(init_lines) > 30:
                    break
            else:
                break
    
    if init_lines:
        print("\n📋 First 30 lines of __init__:\n")
        for line_num, line in init_lines:
            print(f"{line_num:4d} | {line}")
    
else:
    print(f"❌ File not found: {code_path}")

print("\n" + "=" * 70)
