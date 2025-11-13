# train_maddpg_FINAL_FIXED_V4.py
"""
اسکریپت نهایی آموزش MADDPG با ReplayBufferWrapper
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent
from replay_buffer_wrapper import ReplayBufferWrapper

def train_maddpg():
    """
    آموزش MADDPG با wrapper
    """
    print("="*70)
    print("🚀 شروع آموزش MADDPG با ReplayBufferWrapper")
    print("="*70)
    
    # ==================== تنظیمات ====================
    # محیط
    n_agents = 3
    n_users = 5
    dt = 0.1
    area_size = 1000.0
    
    # شبکه
    state_dim = 38  # از env
    action_dim = 4  # (vx, vy, vz, power)
    
    # یادگیری
    lr = 1e-4
    gamma = 0.99
    buffer_size = 1000000
    batch_size = 128
    
    # آموزش
    n_episodes = 1000
    max_steps = 200
    start_training = 1000  # شروع آموزش بعد از این تعداد step
    train_interval = 10    # آموزش هر 10 step
    
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
    print(f"   State dim: {state_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   N agents: {n_agents}")
    
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
    
    # انتقال به device
    agent.actor.to(device)
    agent.critic.to(device)
    agent.actor_target.to(device)
    agent.critic_target.to(device)
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
        # Reset
        states = env.reset()  # (n_agents, state_dim)
        states = np.array(states)
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            total_steps += 1
            
            # Action selection
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states).to(device)  # (n_agents, state_dim)
                
                # انتخاب action برای هر agent
                actions = []
                for i in range(n_agents):
                    state_i = states_tensor[i:i+1]  # (1, state_dim)
                    action_i = agent.act(state_i, noise_scale=0.1)  # (1, action_dim)
                    actions.append(action_i.cpu().numpy())
                
                actions = np.vstack(actions)  # (n_agents, action_dim)
            
            # Step در محیط
            next_states, rewards, dones, info = env.step(actions)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
            dones = np.array(dones)
            
            # ذخیره در buffer (wrapper flatten می‌کند)
            replay_buffer.add(
                states.flatten(),      # (n_agents * state_dim,)
                actions,               # (n_agents, action_dim) -> wrapper flatten می‌کند
                rewards,               # (n_agents,)
                next_states.flatten(), # (n_agents * state_dim,)
                dones                  # (n_agents,)
            )
            
            # آموزش
            if total_steps > start_training and total_steps % train_interval == 0:
                if len(replay_buffer) >= batch_size:
                    # Sample از buffer
                    batch = replay_buffer.sample()
                    states_b, actions_b, rewards_b, next_states_b, dones_b = batch
                    
                    # انتقال به device
                    states_b = states_b.to(device)
                    actions_b = actions_b.to(device)
                    rewards_b = rewards_b.to(device)
                    next_states_b = next_states_b.to(device)
                    dones_b = dones_b.to(device)
                    
                    # Update
                    try:
                        losses = agent.update(
                            replay_buffer,
                            other_agents=None,  # در حالت centralized نیاز نیست
                            batch_size=batch_size
                        )
                        
                        if losses:
                            actor_losses.append(losses.get('actor_loss', 0))
                            critic_losses.append(losses.get('critic_loss', 0))
                    
                    except Exception as e:
                        print(f"\n⚠️  خطا در update (episode {episode}, step {step}): {e}")
            
            # بروزرسانی
            episode_reward += rewards.sum()
            episode_length += 1
            states = next_states
            
            # Done
            if dones.all():
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
    
    # Rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Actor loss
    if actor_losses:
        axes[1, 0].plot(actor_losses)
        axes[1, 0].set_title('Actor Loss')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # Critic loss
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
