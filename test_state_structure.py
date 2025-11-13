# test_state_structure.py

from core.env_multi import MultiUAVEnv

env = MultiUAVEnv(n_agents=3, n_users=5)
state_dict = env.reset()

print("نوع state_dict:", type(state_dict))
print("\nکلیدهای state_dict:", state_dict.keys() if isinstance(state_dict, dict) else "Not a dict!")
print("\nمحتوای کامل state_dict:")
print(state_dict)
