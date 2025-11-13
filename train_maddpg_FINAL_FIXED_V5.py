"""
اسکریپت نهایی آموزش MADDPG با ReplayBufferWrapper
ویرایش V6: پشتیبانی از state dictionary
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent
from replay_buffer_wrapper import ReplayBufferWrapper


def state_dict_to_vector(state_dict):
    """
    تبدیل دیکشنری حالت به بردار مسطح برای هر agent
    
    State structure:
        - uav_positions: (n_agents, 2) 
        - uav_velocities: (n_agents,)
        - uav_angles: (n_agents,)
        - user_positions: (n_users, 2)
        - energy: (n_agents,)
        - distances: (n_agents,)
    
    Returns:
        state_vector: (n_agents, state_dim) numpy array
    """
    n_agents = state_dict['uav_positions'].shape[0]
    n_users = state_dict['user_positions'].shape[0]
    
    state_vectors = []
    
    for i in range(n_agents):
        # State برای agent i
        agent_state = np.concatenate([
            state_dict['uav_positions'][i],      # 2 dims
            [state_dict['uav_velocities'][i]],   # 1 dim
            [state_dict['uav_angles'][i]],       # 1 dim
            state_dict['user_positions'].flatten(),  # n_users*2 dims
            [state_dict['energy'][i]],           # 1 dim
            state_dict['distances']              # n_agents dims
        ])
        
        state_vectors.append(agent_state)
    
    return np.array(state_vectors, dtype=np.float32)


def train_maddpg():
    """
    آموزش MADDPG با wrapper و state dictionary handling
    """
    print("="*70)
    print("🚀 شروع آموزش MADDPG با ReplayBufferWrapper (V6)")
    print("="*70)
    
    # ==================== تنظیمات ====================
    # محیط
    n_agents = 3
    n_users = 5
    dt = 0.1
    area_size = 1000.0
    
    # شبکه
    # state_dim = 2 + 1 + 1 + (n_users*2) + 1 + n_agents
    # state_dim = 2 + 1 + 1 + 10 + 1 + 3 = 18
    state_dim = 2 + 1 + 1 + (n_users * 2) + 1 + n_agents
    action_dim = 4
    
    # یادگیری
    lr = 1e-4
    gamma = 0.99
    buffer_size = 100000
    batch_size = 128
    
    # آموزش
    n_episodes = 1000
    max_steps = 200
    start_training = 500
    train_interval = 10
    
    # ذخیره
    save_interval = 100
    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # ==================== محیط ====================
    print(f"\n🌍 ساخت محیط...")
    env = MultiUAVEnv(
        n_agents=n_agents,
        n_users=n_users,
        dt=dt,
        area_size=area_size
    )
    print(f"   ✓ محیط ساخته شد")
    print(f"   State dim (calculated): {state_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   N agents: {n_agents}")
    print(f"   N users: {n_users}")
    
    # ==================== Agent ====================
    print(f"\n🤖 ساخت Agent...")
    agent = MADDPG_Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        lr=lr,
        gamma=gamma
    )
    print(f"   ✓ Agent ساخته شد")
    
    # انتقال networks به device
    agent.actor.to(device)
    agent.critic.to(device)
    agent.target_actor.to(device)
    agent.target_critic.to(device)
    print(f"   ✓ Networks به {device} منتقل شدند")
    
    # ==================== Replay Buffer (با Wrapper) ====================
    print(f"\n💾 ساخت Replay Buffer با Wrapper...")
    replay_buffer = ReplayBufferWrapper(
        buffer_size=buffer_size,
        batch_size=batch_size,
        n_agents=n_agents,
        action_dim=action_dim
    )
    print(f"   ✓ Buffer ساخته شد")
    print(f"   Buffer size: {buffer_size}")
    print(f"   Batch size: {batch_size}")
    
    # ==================== متریک‌ها ====================
    episode_rewards = []
    episode_lengths = []
    actor_losses = []
    critic_losses = []
    
    total_steps = 0
    
    # ==================== حلقه آموزش ====================
    print(f"\n🎓 شروع آموزش...")
    print(f"   Episodes: {n_episodes}")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   Start training after: {start_training} steps")
    
    for episode in range(n_episodes):
        # Reset - دریافت state dictionary
        state_dict = env.reset()
        states = state_dict_to_vector(state_dict)  # ✅ تبدیل به vector
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            total_steps += 1
            
            # Action selection
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states).to(device)
                
                actions = []
                for i in range(n_agents):
                    state_i = states_tensor[i:i+1]
                    action_i = agent.act(state_i, noise_scale=0.1)
                    actions.append(action_i.cpu().numpy())
                
                actions = np.vstack(actions)
            
            # Step
            next_state_dict, rewards, dones, info = env.step(actions)
            next_states = state_dict_to_vector(next_state_dict)  # ✅ تبدیل به vector
            
            rewards = np.array(rewards, dtype=np.float32)
            dones_array = np.array([dones] * n_agents, dtype=bool)  # ✅ برای هر agent
            
            # ذخیره در buffer
            replay_buffer.add(
                states.flatten(),
                actions,
                rewards,
                next_states.flatten(),
                dones_array
            )
            
            # آموزش
            if total_steps > start_training and total_steps % train_interval == 0:
                if len(replay_buffer) >= batch_size:
                    try:
                        losses = agent.update(
                            replay_buffer,
                            other_agents=None,
                            batch_size=batch_size
                        )
                        
                        if losses:
                            actor_losses.append(losses.get('actor_loss', 0))
                            critic_losses.append(losses.get('critic_loss', 0))
                    
                    except Exception as e:
                        print(f"\n⚠️  خطا در update (episode {episode}, step {step}): {e}")
                        import traceback
                        traceback.print_exc()
            
            # بروزرسانی
            episode_reward += rewards.sum()
            episode_length += 1
            states = next_states
            
            if dones:
                break
        
        # ذخیره متریک‌ها
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # نمایش پیشرفت
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Length: {episode_length} | "
                  f"Buffer: {len(replay_buffer)}")
        
        # ذخیره مدل
        if (episode + 1) % save_interval == 0:
            torch.save(agent.actor.state_dict(), 
                      models_dir / f"actor_episode_{episode+1}.pt")
            torch.save(agent.critic.state_dict(), 
                      models_dir / f"critic_episode_{episode+1}.pt")
            print(f"   💾 Models saved at episode {episode+1}")
    
    # ==================== ذخیره نهایی ====================
    print(f"\n💾 ذخیره مدل‌های نهایی...")
    torch.save(agent.actor.state_dict(), models_dir / "actor_final.pt")
    torch.save(agent.critic.state_dict(), models_dir / "critic_final.pt")
    
    # ذخیره متریک‌ها
    np.savez(results_dir / "training_metrics.npz",
             episode_rewards=episode_rewards,
             episode_lengths=episode_lengths,
             actor_losses=actor_losses,
             critic_losses=critic_losses)
    
    # رسم نمودارها
    print(f"\n📊 رسم نمودارها...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(episode_lengths)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    if actor_losses:
        axes[1, 0].plot(actor_losses)
        axes[1, 0].set_title('Actor Loss')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    if critic_losses:
        axes[1, 1].plot(critic_losses)
        axes[1, 1].set_title('Critic Loss')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=300)
    print(f"   ✓ نمودارها ذخیره شدند")
    
    print("\n" + "="*70)
    print("✅ آموزش تمام شد!")
    print("="*70)
    print(f"📊 Final Stats:")
    print(f"   Total episodes: {n_episodes}")
    print(f"   Total steps: {total_steps}")
    print(f"   Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"   Buffer size: {len(replay_buffer)}")
    print("="*70)


if __name__ == "__main__":
    train_maddpg()
