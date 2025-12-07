# check_agent_signature.py
import sys
sys.path.append('agents')
import inspect

print("="*70)
print("🔍 Checking MADDPGAgent Signature")
print("="*70)

try:
    from maddpg_agent import MADDPGAgent
    
    # بررسی signature
    sig = inspect.signature(MADDPGAgent.__init__)
    print(f"\n📋 MADDPGAgent.__init__() signature:")
    print(f"   {sig}")
    
    # لیست پارامترها
    print(f"\n📝 Parameters:")
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        default = param.default if param.default != inspect.Parameter.empty else "Required"
        print(f"   - {param_name}: {default}")
    
    # بررسی source code
    print(f"\n📄 Source code location:")
    print(f"   {inspect.getfile(MADDPGAgent)}")
    
    # خواندن کد __init__
    import os
    agent_file = "agents/maddpg_agent.py"
    if os.path.exists(agent_file):
        print(f"\n📖 Reading __init__ method...")
        with open(agent_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        in_init = False
        init_lines = []
        indent_level = 0
        
        for line in lines:
            if 'def __init__' in line:
                in_init = True
                indent_level = len(line) - len(line.lstrip())
            
            if in_init:
                init_lines.append(line.rstrip())
                
                # توقف در method بعدی
                if line.strip() and not line.strip().startswith('#'):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= indent_level and len(init_lines) > 1 and 'def ' in line:
                        init_lines = init_lines[:-1]
                        break
        
        if init_lines:
            print("\n" + "─"*70)
            for line in init_lines[:30]:  # اولین 30 خط
                print(line)
            if len(init_lines) > 30:
                print(f"\n   ... ({len(init_lines) - 30} more lines)")
            print("─"*70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
