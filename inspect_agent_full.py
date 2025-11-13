# inspect_agent_full.py
import inspect
from agents.agent_maddpg_multi import MADDPG_Agent

print("="*70)
print("🔍 بررسی کامل کلاس MADDPG_Agent")
print("="*70)

# 1. Signature __init__
sig = inspect.signature(MADDPG_Agent.__init__)
print("\n📋 پارامترهای __init__:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        default = param.default if param.default != inspect.Parameter.empty else "⚠️ الزامی"
        print(f"   {param_name}: {default}")

# 2. Signature متد act
try:
    act_sig = inspect.signature(MADDPG_Agent.act)
    print("\n🎬 پارامترهای متد act:")
    for param_name, param in act_sig.parameters.items():
        if param_name != 'self':
            default = param.default if param.default != inspect.Parameter.empty else "⚠️ الزامی"
            print(f"   {param_name}: {default}")
except Exception as e:
    print(f"\n⚠️ نمی‌توان signature متد act را بررسی کرد: {e}")

# 3. Signature متد update
try:
    update_sig = inspect.signature(MADDPG_Agent.update)
    print("\n🔄 پارامترهای متد update:")
    for param_name, param in update_sig.parameters.items():
        if param_name != 'self':
            default = param.default if param.default != inspect.Parameter.empty else "⚠️ الزامی"
            print(f"   {param_name}: {default}")
except Exception as e:
    print(f"\n⚠️ نمی‌توان signature متد update را بررسی کرد: {e}")

# 4. Attributes
print("\n📦 ویژگی‌های Agent (پس از ایجاد):")
try:
    agent = MADDPG_Agent(state_dim=38, action_dim=4, n_agents=3, lr=0.001, gamma=0.99)
    
    attrs = [attr for attr in dir(agent) if not attr.startswith('_')]
    print(f"   تعداد کل: {len(attrs)}")
    print(f"   لیست: {attrs[:10]}...")
    
    # بررسی وجود actors یا actor
    if hasattr(agent, 'actors'):
        print(f"\n   ✅ agent.actors وجود دارد (تعداد: {len(agent.actors)})")
    elif hasattr(agent, 'actor'):
        print(f"\n   ✅ agent.actor وجود دارد (نوع: {type(agent.actor)})")
        # آیا actor یک لیست است؟
        if isinstance(agent.actor, list):
            print(f"      📋 actor یک لیست است با {len(agent.actor)} عضو")
    
    # بررسی critics
    if hasattr(agent, 'critics'):
        print(f"   ✅ agent.critics وجود دارد (تعداد: {len(agent.critics)})")
    elif hasattr(agent, 'critic'):
        print(f"   ✅ agent.critic وجود دارد (نوع: {type(agent.critic)})")
        if isinstance(agent.critic, list):
            print(f"      📋 critic یک لیست است با {len(agent.critic)} عضو")
    
    # بررسی replay buffer
    if hasattr(agent, 'memory'):
        print(f"   ✅ agent.memory وجود دارد (نوع: {type(agent.memory)})")
    elif hasattr(agent, 'buffer'):
        print(f"   ✅ agent.buffer وجود دارد (نوع: {type(agent.buffer)})")
    elif hasattr(agent, 'replay_buffer'):
        print(f"   ✅ agent.replay_buffer وجود دارد (نوع: {type(agent.replay_buffer)})")
    
except Exception as e:
    print(f"   ❌ خطا در ایجاد agent: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
