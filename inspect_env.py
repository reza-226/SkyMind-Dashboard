# inspect_env.py
import inspect
from core.env_multi import MultiUAVEnv

print("="*70)
print("🔍 بررسی کلاس MultiUAVEnv")
print("="*70)

# بررسی signature __init__
sig = inspect.signature(MultiUAVEnv.__init__)
print("\n📋 پارامترهای __init__:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        default = param.default if param.default != inspect.Parameter.empty else "⚠️ الزامی"
        print(f"   {param_name}: {default}")

print("\n" + "="*70)

# تست ساخت محیط
print("\n🧪 تست ساخت محیط...")
try:
    # تلاش 1: بدون پارامتر
    print("\n1️⃣ بدون پارامتر:")
    env = MultiUAVEnv()
    print(f"   ✅ موفق!")
    
    # بررسی attributes
    if hasattr(env, 'n_agents'):
        print(f"   📊 n_agents: {env.n_agents}")
    if hasattr(env, 'n_users'):
        print(f"   📊 n_users: {env.n_users}")
    if hasattr(env, 'grid_size'):
        print(f"   📊 grid_size: {env.grid_size}")
    if hasattr(env, 'area_size'):
        print(f"   📊 area_size: {env.area_size}")
        
except Exception as e:
    print(f"   ❌ خطا: {e}")

try:
    # تلاش 2: با n_agents
    print("\n2️⃣ با n_agents=3:")
    env = MultiUAVEnv(n_agents=3)
    print(f"   ✅ موفق!")
    
except Exception as e:
    print(f"   ❌ خطا: {e}")

try:
    # تلاش 3: با n_agents و n_users
    print("\n3️⃣ با n_agents=3, n_users=5:")
    env = MultiUAVEnv(n_agents=3, n_users=5)
    print(f"   ✅ موفق!")
    
except Exception as e:
    print(f"   ❌ خطا: {e}")

print("\n" + "="*70)
