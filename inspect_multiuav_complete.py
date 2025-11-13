# train_maddpg_complete.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent
from stable_baselines3.common.buffers import ReplayBuffer


def state_dict_to_array(state_dict, n_agents):
    """
    تبدیل state_dict به آرایه (n_agents, state_dim_per_agent)
    """
    states = []
    for i in range(n_agents):
        agent_state = state_dict[i]
        state_vec = np.concatenate([
            agent_state['position'],  # [x, y, z]
            [agent_state['energy']],
            [agent_state['queue_size']],
            [agent_state['trust_score']]
        ])
        states.append(state_vec)
    return np.array(states)


def train_maddpg(
    n_episodes=1000,
    max_steps=200,
    batch_size=128,
    buffer_size=100000,
    start_training=500,
    save_freq=100,
    device='cpu'
):
    print("="*70)
    print("🚀 شروع آموزش MADDPG کامل (Multi-Agent)")
    print("="*70)
    print(f"\n🖥️  Device: {device}\n")
    
    # ========== ساخت محیط ==========
    print("🌍 ساخت محیط...")
    env = MultiUAVEnv(n_agents=3, n_users=5)
    n_agents = env.n_agents
    print("   ✓ محیط ساخته شد")
    
    # محاسبه dimensions
    state_dict = env.reset()
    state_array = state_dict_to_array(state_dict, n_agents)
    state_dim_per_agent = state_array.shape[1]  # 6
    action_dim_per_agent = env.action_space.shape[0] // n_agents  # 4
    
    print(f"   N agents: {n_agents}")
    print(f"   State dim per agent: {state_dim_per_agent}")
    print(f"   Action dim per agent: {action_dim_per_agent}")
    print(f"   N users: {env.n_users}\n")
    
    # ========== ساخت Agents ==========
    print("🤖 ساخت Agents...")
    agents = []
    for i in range(n_agents):
        agent = MADDPG_Agent(
            agent_id=i,
            state_dim_per_agent=state_dim_per_agent,
            action_dim_per_agent=action_dim_per_agent,
            n_agents=n_agents,
            hidden_dim=256,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.95,
            tau=0.01,
            device=device
        )
        agents.append(agent)
    print(f"   ✓ {n_agents} Agents ساخته شدند\n")
    
    # ========== ساخت Replay Buffer ==========
    print("💾 ساخت Replay Buffer...")
    # برای MADDPG، buffer باید states و actions تمام agents را ذخیره کند
    # شکل: (n_agents, dim)
    total_state_dim = state_dim_per_agent * n_agents
    total_action_dim = action_dim_per_agent * n_agents
    
    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        n_envs=1
    )
    print(f"   ✓ Buffer ساخته شد (size: {buffer_size})\n")
    
    # ========== آموزش ==========
    print("🎓 شروع آموزش...")
    print(f"   Episodes: {n_episodes}")
    print(f"   Max steps: {max_steps}")
    print(f"   Start training after: {start_training} steps\n")
    
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    
    for episode in range(1, n_episodes + 1):
        state_dict = env.reset()
        states = state_dict_to_array(state_dict, n_agents)  # (n_agents, state_dim)
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # هر agent action خودش را انتخاب می‌کند
            actions = []
            noise_scale = max(0.1, 1.0 - episode / (n_episodes * 0.5))
            
            for i, agent in enumerate(agents):
                agent_state = states[i]
                action = agent.act(agent_state, noise_scale=noise_scale)
                actions.append(action)
            
            actions = np.array(actions)  # (n_agents, action_dim)
            actions_flat = actions.flatten()  # برای env.step()
            
            # اجرای action
            next_state_dict, reward, done, info = env.step(actions_flat)
            next_states = state_dict_to_array(next_state_dict, n_agents)
            
            # ذخیره در buffer
            # buffer می‌خواهد: obs, next_obs, action, reward, done
            replay_buffer.add(
                obs=states.flatten(),
                next_obs=next_states.flatten(),
                action=actions_flat,
                reward=reward,
                done=done,
                infos=[info]
            )
            
            # آموزش هر agent
            if total_steps >= start_training:
                for agent in agents:
                    losses = agent.update(replay_buffer, batch_size, agents)
            
            # آماده برای step بعدی
            states = next_states
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # گزارش
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}/{n_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg(10): {avg_reward:.2f} | "
                  f"Length: {episode_length} | "
                  f"Buffer: {replay_buffer.size()}")
        
        # ذخیره models
        if episode % save_freq == 0:
            for i, agent in enumerate(agents):
                agent.save(f"models/maddpg_agent{i}_ep{episode}.pt")
    
    # ذخیره نهایی
    for i, agent in enumerate(agents):
        agent.save(f"models/maddpg_agent{i}_final.pt")
    print("\n✓ Models ذخیره شدند")
    
    # رسم نتایج
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards (MADDPG)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('results/maddpg_training.png')
    print("✓ نمودار ذخیره شد: results/maddpg_training.png")
    
    return agents, episode_rewards, episode_lengths


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_maddpg(device=device)
