# check_agent_params.py
import sys
sys.path.append('environments')
sys.path.append('models/actor_critic')

from maddpg_agent import MADDPGAgent
import inspect

print("="*60)
print("🔍 Checking MADDPGAgent Parameters")
print("="*60)

sig = inspect.signature(MADDPGAgent.__init__)
print(f"\n📋 __init__ signature:")
print(f"   {sig}")

print("\n📝 Parameters:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        default = param.default if param.default != inspect.Parameter.empty else "Required"
        print(f"   • {param_name}: {default}")

print("\n" + "="*60)
