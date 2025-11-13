"""
test_baseline.py - مقایسه با استراتژی All-Local
"""

def test_baseline_all_local(num_episodes=100):
    """همه وظایف محلی پردازش شوند"""
    env = MultiUAVEnv(n_agents=3, n_users=5)
    
    baseline_rewards = []
    baseline_energies = []
    baseline_delays = []
    
    for ep in tqdm(range(num_episodes), desc="Baseline"):
        state = env.reset()
        episode_reward = 0
        
        for step in range(200):
            # اکشن صفر = پردازش محلی
            actions = np.zeros((3, 4))
            
            next_state, reward, done, info = env.step(actions)
            
            if isinstance(reward, np.ndarray):
                reward = reward.sum()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        baseline_rewards.append(episode_reward)
        baseline_energies.append(info.get('energy_total', 0))
        baseline_delays.append(info.get('mean_delay', 0))
    
    print("\n📊 نتایج Baseline (All-Local):")
    print(f"Mean Reward: {np.mean(baseline_rewards):.4f}")
    print(f"Mean Energy: {np.mean(baseline_energies):.4f}")
    print(f"Mean Delay:  {np.mean(baseline_delays):.4f}")
