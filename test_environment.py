# test_environment.py
from environment import UAVEnvironment

env = UAVEnvironment(num_tasks=10, task_complexity='mixed')

# در Gymnasium، reset() یک tuple برمی‌گردونه: (observation, info)
observation, info = env.reset()

print(f"✅ State shape: {observation.shape}")
print(f"✅ Observation space: {env.observation_space}")
print(f"✅ Action space: {env.action_space}")
print(f"✅ Reset info: {info}")

# Test one step
action = env.action_space.sample()
next_state, reward, terminated, truncated, info = env.step(action)

print(f"\n✅ Action shape: {action.shape}")
print(f"✅ Next state shape: {next_state.shape}")
print(f"✅ Reward: {reward:.2f}")
print(f"✅ Terminated: {terminated}")
print(f"✅ Truncated: {truncated}")
print(f"✅ Info keys: {list(info.keys())}")
print(f"✅ Offload layer: {info['offload_layer']}")
print(f"✅ Success: {info['success']}")

# Test multiple steps
print("\n" + "="*60)
print("🧪 Testing 5 consecutive steps:")
print("="*60)

env.reset()
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: Reward={reward:6.2f} | Layer={info['offload_layer']:6s} | "
          f"Success={info['success']} | Steps={info['step']}")
    
    if terminated or truncated:
        print("Episode finished!")
        break
