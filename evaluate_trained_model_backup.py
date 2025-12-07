#!/usr/bin/env python3
"""
Evaluate MADDPG Curriculum Learning - FIXED Actor Structure
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from pettingzoo.mpe import simple_spread_v3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActorNetwork(torch.nn.Module):
    """شبکه Actor - مطابق با ساختار Sequential در training"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        # استفاده از Sequential با نام 'network'
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),     # 0
            torch.nn.ReLU(),                          # 1
            torch.nn.Linear(hidden_dim, hidden_dim),  # 2
            torch.nn.ReLU(),                          # 3
            torch.nn.Linear(hidden_dim, action_dim),  # 4
            torch.nn.Softmax(dim=-1)                  # 5
        )
        
    def forward(self, x):
        return self.network(x)


def load_actor_from_checkpoint(checkpoint, obs_dim, action_dim):
    """استخراج Actor از checkpoint"""
    
    actor = ActorNetwork(obs_dim, action_dim)
    
    if isinstance(checkpoint, dict) and 'actor' in checkpoint:
        actor_state = checkpoint['actor']
        
        try:
            actor.load_state_dict(actor_state)
            logger.info("✅ Successfully loaded actor weights!")
            return actor
        except Exception as e:
            logger.error(f"❌ Error loading actor: {e}")
            return None
    
    return None


def load_curriculum_models(level="level3_complex"):
    """لود مدل‌های Curriculum Learning"""
    
    model_dir = f"models/{level}/final"
    
    if not os.path.exists(model_dir):
        logger.error(f"❌ Model directory not found: {model_dir}")
        return None
    
    files = os.listdir(model_dir)
    logger.info(f"📁 Files in {model_dir}: {files}")
    
    # ایجاد environment
    env = simple_spread_v3.parallel_env(N=2, local_ratio=0.5, max_cycles=25)
    env.reset()
    
    sample_agent = env.agents[0]
    obs_dim = env.observation_space(sample_agent).shape[0]
    action_dim = env.action_space(sample_agent).n
    
    logger.info(f"📐 Observation: {obs_dim}, Action: {action_dim}")
    
    # لود مدل‌ها
    actors = {}
    
    # فایل مدل اصلی
    model_file = os.path.join(model_dir, 'agent_0.pth')
    
    if not os.path.exists(model_file):
        logger.error(f"❌ Model file not found: {model_file}")
        return None
    
    try:
        checkpoint = torch.load(model_file, map_location='cpu')
        
        # بررسی ساختار
        if 'actor' in checkpoint:
            actor_state = checkpoint['actor']
            logger.info(f"🔑 Actor keys: {list(actor_state.keys())}")
        
        # لود برای هر دو agent
        for agent_name in env.agents:
            actor = load_actor_from_checkpoint(checkpoint, obs_dim, action_dim)
            
            if actor is None:
                logger.error(f"❌ Failed to load actor for {agent_name}")
                return None
            
            actor.eval()
            actors[agent_name] = actor
            logger.info(f"✅ Loaded {agent_name}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return env, actors


def evaluate_models(env, actors, num_episodes=20):
    """ارزیابی مدل‌های لود شده"""
    
    episode_rewards = []
    episode_steps = []
    agent_rewards = {agent: [] for agent in env.agents}
    successful_episodes = 0
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"🎯 Starting Evaluation: {num_episodes} episodes")
    logger.info("=" * 60)
    
    for ep in range(num_episodes):
        observations, _ = env.reset()
        ep_rewards = {agent: 0 for agent in env.agents}
        steps = 0
        
        while True:
            actions = {}
            for agent in env.agents:
                obs = torch.FloatTensor(observations[agent]).unsqueeze(0)
                with torch.no_grad():
                    action_probs = actors[agent](obs)
                    action = torch.argmax(action_probs, dim=-1).item()
                actions[agent] = action
            
            observations, rewards, terminations, truncations, _ = env.step(actions)
            
            for agent in env.agents:
                ep_rewards[agent] += rewards[agent]
            
            steps += 1
            
            if all(terminations.values()) or all(truncations.values()):
                break
        
        avg_reward = np.mean(list(ep_rewards.values()))
        episode_rewards.append(avg_reward)
        episode_steps.append(steps)
        
        if avg_reward > -10:
            successful_episodes += 1
        
        for agent in env.agents:
            agent_rewards[agent].append(ep_rewards[agent])
        
        status = "✅" if avg_reward > -10 else "❌"
        logger.info(f"{status} Episode {ep+1:2d}/{num_episodes} | Reward: {avg_reward:6.2f} | Steps: {steps}")
    
    # خلاصه نتایج
    logger.info("")
    logger.info("=" * 60)
    logger.info("📈 EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mean Reward:    {np.mean(episode_rewards):6.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"Best Reward:    {max(episode_rewards):6.2f}")
    logger.info(f"Worst Reward:   {min(episode_rewards):6.2f}")
    
    success_rate = (successful_episodes / num_episodes) * 100
    logger.info(f"Success Rate:   {success_rate:5.1f}% ({successful_episodes}/{num_episodes})")
    logger.info("=" * 60)
    
    # رسم نمودار
    plot_results(episode_rewards, agent_rewards, episode_steps)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': success_rate,
        'best_reward': max(episode_rewards),
        'worst_reward': min(episode_rewards)
    }


def plot_results(episode_rewards, agent_rewards, episode_steps):
    """رسم نمودارهای نتایج"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MADDPG Curriculum Learning - Evaluation Results', fontsize=14, fontweight='bold')
    
    episodes = range(1, len(episode_rewards)+1)
    
    # 1. Reward per episode
    ax1 = axes[0, 0]
    ax1.plot(episodes, episode_rewards, 'b-o', linewidth=2, markersize=6, label='Episode Reward')
    ax1.axhline(y=-10, color='r', linestyle='--', linewidth=2, label='Success Threshold')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Average Reward', fontsize=11)
    ax1.set_title('Reward per Episode', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward distribution
    ax2 = axes[0, 1]
    ax2.hist(episode_rewards, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=np.mean(episode_rewards), color='green', linestyle='-', linewidth=2, label='Mean')
    ax2.axvline(x=-10, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Reward', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Steps per episode
    ax3 = axes[1, 0]
    ax3.plot(episodes, episode_steps, 'orange', marker='s', linewidth=2, markersize=6)
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Steps', fontsize=11)
    ax3.set_title('Steps per Episode', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Per-agent rewards
    ax4 = axes[1, 1]
    agents = list(agent_rewards.keys())
    means = [np.mean(agent_rewards[a]) for a in agents]
    stds = [np.std(agent_rewards[a]) for a in agents]
    
    x_pos = range(len(agents))
    ax4.bar(x_pos, means, yerr=stds, color=['#2ecc71', '#3498db'], 
            edgecolor='black', capsize=5, alpha=0.8)
    ax4.axhline(y=-10, color='r', linestyle='--', linewidth=2, label='Threshold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(agents)
    ax4.set_ylabel('Average Reward', fontsize=11)
    ax4.set_title('Per-Agent Performance', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    logger.info("📊 Plot saved: evaluation_results.png")
    plt.show()


def main():
    print("\n" + "="*60)
    print("📚 MADDPG Curriculum Learning - Model Evaluation")
    print("="*60)
    print("  1. Level 1 - Simple")
    print("  2. Level 2 - Medium") 
    print("  3. Level 3 - Complex [RECOMMENDED]")
    print("="*60)
    
    levels = {'1': 'level1_simple', '2': 'level2_medium', '3': 'level3_complex'}
    choice = input("\nSelect level (1/2/3) [default: 3]: ").strip() or '3'
    
    if choice not in levels:
        print("❌ Invalid choice!")
        return
    
    selected_level = levels[choice]
    logger.info(f"✅ Selected: {selected_level}\n")
    
    result = load_curriculum_models(selected_level)
    
    if result is None:
        logger.error("❌ Failed to load models!")
        return
    
    env, actors = result
    results = evaluate_models(env, actors, num_episodes=20)
    env.close()
    
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS")
    print("="*60)
    print(f"Mean Reward:  {results['mean_reward']:6.2f}")
    print(f"Success Rate: {results['success_rate']:5.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
