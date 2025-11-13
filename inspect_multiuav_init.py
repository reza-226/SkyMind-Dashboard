"""
inspect_multiuav_init.py
بررسی پارامترهای __init__ در MultiUAVEnv
"""

import sys
import inspect
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.env_multi import MultiUAVEnv

print("=" * 70)
print("🔍 Inspecting MultiUAVEnv.__init__")
print("=" * 70)

# استخراج signature
sig = inspect.signature(MultiUAVEnv.__init__)

print("\n📋 __init__ Parameters:")
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    
    default = param.default
    if default == inspect.Parameter.empty:
        default_str = "REQUIRED"
    else:
        default_str = f"= {default}"
    
    print(f"   {param_name}: {default_str}")

# خواندن کد __init__
print("\n" + "=" * 70)
print("📄 Reading __init__ code")
print("=" * 70)

code_path = Path("core/env_multi.py")
if code_path.exists():
    with open(code_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_init = False
    init_lines = []
    indent_level = 0
    
    for i, line in enumerate(lines, 1):
        if 'def __init__(' in line:
            in_init = True
            indent_level = len(line) - len(line.lstrip())
            init_lines.append((i, line.rstrip()))
        elif in_init:
            current_indent = len(line) - len(line.lstrip())
            
            if line.strip() == '' or current_indent > indent_level:
                init_lines.append((i, line.rstrip()))
                
                # فقط 40 خط اول
                if len(init_lines) > 40:
                    break
            else:
                break
    
    if init_lines:
        print("\n📋 First 40 lines of __init__:\n")
        for line_num, line in init_lines:
            print(f"{line_num:4d} | {line}")
else:
    print(f"❌ File not found: {code_path}")

print("\n" + "=" * 70)
