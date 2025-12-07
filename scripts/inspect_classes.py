# scripts/inspect_classes.py
"""
🔍 ابزار بررسی کلاس‌ها و متدها
استفاده: python scripts/inspect_classes.py
"""

import inspect
import importlib
from pathlib import Path

def inspect_module(module_path):
    """بررسی یک ماژول و نمایش کلاس‌ها و متدها"""
    try:
        # Import module
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"\n{'='*80}")
        print(f"📂 File: {module_path}")
        print(f"{'='*80}")
        
        # Get all classes
        classes = []
        functions = []
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                classes.append((name, obj))
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                functions.append((name, obj))
        
        # Print classes
        if classes:
            print(f"\n🎯 Classes ({len(classes)}):")
            for name, cls in classes:
                print(f"   ├─ {name}")
                
                # Get methods
                methods = [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m))]
                if methods:
                    for method in methods[:5]:  # Show first 5
                        print(f"   │  ├─ {method}()")
                    if len(methods) > 5:
                        print(f"   │  └─ ... ({len(methods)-5} more)")
        
        # Print functions
        if functions:
            print(f"\n⚙️  Functions ({len(functions)}):")
            for name, func in functions:
                sig = inspect.signature(func)
                print(f"   ├─ {name}{sig}")
        
        return classes, functions
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return [], []


def main():
    """بررسی تمام فایل‌های مهم"""
    
    files_to_check = [
        "algorithms/baselines/simple_policies.py",
        "algorithms/baselines/dqn_agent.py",
        "algorithms/baselines/ddpg_agent.py",
        "core/env/environment.py",
        "agents/maddpg_agent.py",
    ]
    
    print("🔍 CLASS & METHOD INSPECTOR")
    print("="*80)
    
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            inspect_module(path)
        else:
            print(f"\n⚠️  File not found: {file_path}")
    
    print("\n" + "="*80)
    print("✅ Inspection Complete!")


if __name__ == "__main__":
    main()
