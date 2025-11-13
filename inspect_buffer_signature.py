# inspect_buffer_signature.py
"""
بررسی امضای __init__ کلاس ReplayBuffer
"""

import inspect
from agents.agent_maddpg_multi import ReplayBuffer

print("="*70)
print("🔍 بررسی ReplayBuffer.__init__ signature")
print("="*70)

try:
    # دریافت signature
    sig = inspect.signature(ReplayBuffer.__init__)
    
    print(f"\n📋 امضای کامل:")
    print(f"   {sig}")
    
    print(f"\n📝 پارامترها:")
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
            
        default = param.default
        if default == inspect.Parameter.empty:
            default_str = "(required)"
        else:
            default_str = f"= {default}"
        
        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            type_str = ""
        else:
            type_str = f": {annotation}"
            
        print(f"   • {param_name}{type_str} {default_str}")
    
    # نگاه کردن به کد منبع
    print(f"\n📄 کد منبع __init__:")
    source = inspect.getsource(ReplayBuffer.__init__)
    # چاپ 20 خط اول
    lines = source.split('\n')[:20]
    for i, line in enumerate(lines, 1):
        print(f"   {i:2d}: {line}")
    
except Exception as e:
    print(f"\n❌ خطا: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
