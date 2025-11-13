"""
اسکریپت نهایی آموزش MADDPG با ReplayBufferWrapper
ویرایش V7: اصلاح state_dict_to_vector
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
    
    State structure from env:
        - uav_positions: (n_agents, 2) 
        - uav_velocities: (n_agents,)
        - uav_angles: (n_agents,)
        - user_positions: (n_users, 2)
        - energy: (n_agents,)
        - distances: (n_agents,)
    
    ✅ هر agent فقط state خودش را می‌بیند:
        [pos_x, pos_y, velocity, angle, energy, avg_distance_to_users]
    
    Returns:
        state_vector: (n_agents, state_dim_per_agent) numpy array
        که state_dim_per_agent = 6
    """
    n_agents = state_dict['uav_positions'].shape[0]
    
    state_vectors = []
    
    for i in range(n_agents):
        # State برای agent i - فقط اطلاعات خودش
        agent_state = np.array([
            state_dict['uav_positions'][i, 0],    # pos_x
            state_dict['uav_positions'][i, 1],    # pos_y
            state_dict['uav_velocities'][i],      # velocity
            state_dict['uav_angles'][i],          # angle
            state_dict['energy'][i],              # energy
            state_dict['distances'][i]            # avg distance to users
        ], dtype=np.float32)
        
        state_vectors.append(agent_state)
    
    return np.array(state_vectors, dtype=np.float32)  # shape: (n_agents, 6)


def train_maddpg():
    """
    آموزش MADDPG با wrapper و state dictionary handling
    """
    print("="*70)
    print("🚀 شروع آموزش MADDPG با ReplayBufferWrapper (V7 - Fixed)")
    print("="*70)
    
    # ==================== تنظیمات ====================
    # محیط
    n_agents = 3
    n_users = 5
    dt = 0.1
    area_size = 1000.0
    
    # شبکه - ✅ اصلاح شده
    state_dim = 6  # [pos_x, pos_y, velocity, angle, energy, distance]
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
    
    # ✅ بررسی state_dim واقعی
    test_state_dict = env.reset()
    test_state_vector = state_dict_to_vector(test_state_dict)
    actual_state_dim = test_state_vector.shape[1]
    
    print(f"   ✓ محیط ساخته شد")
    print(f"   State dim per agent: {actual_state_dim}")
    print(f"   Action dim per agent: {action_dim}")
    print(f"   N agents: {n_agents}")
    print(f"   N users: {n_users}")
    print(f"   Total state dim (flattened): {actual_state_dim * n_agents}")
    print(f"   State dict keys: {list(test_state_dict.keys())}")
    
    # بروزرسانی state_dim
    state_dim = actual_state_dim
    
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
    print(f"   Train interval: every {train_interval} steps")
    
    for episode in range(n_episodes):
        # Reset - دریافت state dictionary
        state_dict = env.reset()
        states = state_dict_to_vector(state_dict)  # ✅ shape: (n_agents, state_dim)
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            total_steps += 1
            
            # Action selection
            actions = []
            noise_scale = max(0.1, 1.0 - episode / (n_episodes * 0.5))  # decay noise
            
            with torch.no_grad():
                for i in range(n_agents):
                    state_i = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
                    action_i = agent.act(state_i, noise_scale=noise_scale)
                    actions.append(action_i)  # ✅ مستقیماً numpy array است
            
            actions = np.vstack(actions)  # shape: (n_agents, action_dim)
            
            # Step
            next_state_dict, rewards, dones, info = env.step(actions)
            next_states = state_dict_to_vector(next_state_dict)  # ✅ shape: (n_agents, state_dim)
            
            # تبدیل rewards
            if isinstance(rewards, (int, float)):
                rewards = np.array([rewards] * n_agents, dtype=np.float32)
            else:
                rewards = np.array(rewards, dtype=np.float32)
            
            # تبدیل dones
            if isinstance(dones, bool):
                dones_array = np.array([dones] * n_agents, dtype=bool)
            else:
                dones_array = np.array(dones, dtype=bool)
            
            # ذخیره در buffer
            replay_buffer.add(
                states.flatten(),        # flatten to 1D
                actions,                 # (n_agents, action_dim)
                rewards,                 # (n_agents,)
                next_states.flatten(),   # flatten to 1D
                dones_array              # (n_agents,)
            )
            
            # آموزش
            if total_steps > start_training and total_steps % train_interval == 0:
                if len(replay_buffer) >= batch_size:
                    try:
                        losses = agent.update(
                            replay_buffer,
                            other_agents=[],  # ✅ لیست خالی برای single-agent training
                            batch_size=batch_size
                        )
                        
                        if losses:
                            actor_losses.append(losses.get('actor_loss', 0))
                            critic_losses.append(losses.get('critic_loss', 0))
                    
                    except Exception as e:
                        print(f"\n⚠️  خطا در update (episode {episode+1}, step {step}): {e}")
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
            avg_actor_loss = np.mean(actor_losses[-100:]) if actor_losses else 0
            avg_critic_loss = np.mean(critic_losses[-100:]) if critic_losses else 0
            
            print(f"Episode {episode+1:4d}/{n_episodes} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(10): {avg_reward:7.2f} | "
                  f"Length: {episode_length:3d} | "
                  f"Buffer: {len(replay_buffer):6d} | "
                  f"A_Loss: {avg_actor_loss:.4f} | "
                  f"C_Loss: {avg_critic_loss:.4f}")
        
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
    print(f"   ✓ Models ذخیره شدند")
    
    # ذخیره متریک‌ها
    np.savez(results_dir / "training_metrics.npz",
             episode_rewards=episode_rewards,
             episode_lengths=episode_lengths,
             actor_losses=actor_losses,
             critic_losses=critic_losses)
    print(f"   ✓ Metrics ذخیره شدند")
    
    # رسم نمودارها
    print(f"\n📊 رسم نمودارها...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode Rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, label='Raw')
    if len(episode_rewards) >= 50:
        window = 50
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                       'r-', linewidth=2, label=f'MA({window})')
    axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode Lengths
    axes[0, 1].plot(episode_lengths, alpha=0.6)
    if len(episode_lengths) >= 50:
        window = 50
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_lengths)), moving_avg, 
                       'g-', linewidth=2, label=f'MA({window})')
    axes[0, 1].set_title('Episode Lengths', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Actor Loss
    if actor_losses:
        axes[1, 0].plot(actor_losses, alpha=0.4)
        if len(actor_losses) >= 100:
            window = 100
            moving_avg = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(actor_losses)), moving_avg, 
                           'r-', linewidth=2, label=f'MA({window})')
        axes[1, 0].set_title('Actor Loss', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Critic Loss
    if critic_losses:
        axes[1, 1].plot(critic_losses, alpha=0.4)
        if len(critic_losses) >= 100:
            window = 100
            moving_avg = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(critic_losses)), moving_avg, 
                           'b-', linewidth=2, label=f'MA({window})')
        axes[1, 1].set_title('Critic Loss', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ نمودارها ذخیره شدند: {results_dir / 'training_curves.png'}")
    
    print("\n" + "="*70)
    print("✅ آموزش تمام شد!")
    print("="*70)
    print(f"📊 Final Statistics:")
    print(f"   Total episodes: {n_episodes}")
    print(f"   Total steps: {total_steps}")
    print(f"   Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"   Max reward: {max(episode_rewards):.2f} (Episode {np.argmax(episode_rewards)+1})")
    print(f"   Buffer size: {len(replay_buffer)}")
    if actor_losses:
        print(f"   Final actor loss: {np.mean(actor_losses[-100:]):.4f}")
    if critic_losses:
        print(f"   Final critic loss: {np.mean(critic_losses[-100:]):.4f}")
    print("="*70)


if __name__ == "__main__":
    train_maddpg()
