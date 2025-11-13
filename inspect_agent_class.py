# inspect_agent_class.py
import sys
sys.path.append('.')

print("="*60)
print("🔍 بررسی کلاس‌های Agent در agent_maddpg_multi.py")
print("="*60)

try:
    import agents.agent_maddpg_multi as agent_module
    
    print("\n✅ ماژول import شد")
    print(f"\nکلاس‌های موجود:")
    
    classes = [name for name in dir(agent_module) 
               if not name.startswith('_') and isinstance(getattr(agent_module, name), type)]
    
    if classes:
        for cls_name in classes:
            cls = getattr(agent_module, cls_name)
            print(f"\n  📦 {cls_name}")
            
            # بررسی متدها
            methods = [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m))]
            if methods:
                print(f"     متدها: {', '.join(methods[:5])}")
                if len(methods) > 5:
                    print(f"            + {len(methods)-5} متد دیگر")
    else:
        print("  ⚠️ هیچ کلاسی پیدا نشد")
    
    print("\n" + "="*60)
    print("💡 برای import صحیح از این نام استفاده کنید:")
    if classes:
        print(f"   from agents.agent_maddpg_multi import {classes[0]}")
    print("="*60)
    
except ImportError as e:
    print(f"\n❌ خطای Import: {e}")
    
except Exception as e:
    print(f"\n❌ خطا: {e}")
    import traceback
    traceback.print_exc()
