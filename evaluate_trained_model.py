"""
Evaluate MADDPG - Level 3 with Adversary Configuration
"""

import torch
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ActorNetwork(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

def create_adversary_environment():
    """Create simple_spread with adversary configuration"""
    from pettingzoo.mpe import simple_adversary_v3
    
    logger.info("   Creating simple_adversary_v3 environment")
    logger.info("   Config: N=2 good agents, 1 adversary, 2 landmarks")
    
    env = simple_adversary_v3.env(
        N=2,  # 2 good agents
        continuous_actions=True,
        max_cycles=25,
        render_mode=None
    )
    
    return env

def load_agent(checkpoint_path, obs_dim, action_dim, hidden_dim):
    """Load agent from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        actor = ActorNetwork(obs_dim, action_dim, hidden_dim)
        actor.load_state_dict(checkpoint['actor'])
        actor.eval()
        
        logger.info(f"   ✓ Loaded: {checkpoint_path.name} (obs={obs_dim})")
        return actor
        
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return None

def evaluate_episode(env, agents_dict, max_steps=50):
    """Run one evaluation episode"""
    env.reset()
    episode_rewards = {agent: 0.0 for agent in env.agents}
    step_count = 0
    
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        episode_rewards[agent_name] += reward
        step_count += 1
        
        if termination or truncation:
            action = None
        else:
            actor = agents_dict.get(agent_name)
            
            if actor is None:
                action = env.action_space(agent_name).sample()
            else:
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                with torch.no_grad():
                    action_raw = actor(obs_tensor).squeeze(0).numpy()
                
                # Convert from [-1, 1] to [0, 1]
                action = (action_raw + 1.0) / 2.0
                action = np.clip(action, 0.0, 1.0)
        
        env.step(action)
        
        if step_count >= max_steps * len(env.agents):
            break
    
    total_reward = sum(episode_rewards.values())
    return total_reward, episode_rewards, step_count // len(env.agents)

def main():
    logger.info("=" * 80)
    logger.info("🎯 MADDPG Evaluation - Level 3 (Adversary Environment)")
    logger.info("=" * 80)
    
    MODEL_DIR = Path("models/level3_complex/final")
    NUM_EVAL_EPISODES = 100
    
    # Create environment
    logger.info("\n📦 Creating environment...")
    env = create_adversary_environment()
    env.reset()
    
    logger.info(f"   ✓ Agents: {env.agents}")
    for agent in env.agents:
        obs_shape = env.observation_space(agent).shape[0]
        act_shape = env.action_space(agent).shape[0]
        logger.info(f"   {agent}: obs={obs_shape}, act={act_shape}")
    
    # Load models
    logger.info("\n🤖 Loading models...")
    
    agents_dict = {}
    
    # Load good agents (agent_0, agent_1)
    agent_0_path = MODEL_DIR / "agent_0.pth"
    if agent_0_path.exists():
        actor = load_agent(agent_0_path, obs_dim=14, action_dim=5, hidden_dim=128)
        if actor:
            agents_dict['agent_0'] = actor
            agents_dict['agent_1'] = actor  # Share same policy
    
    # Load adversary
    adversary_path = MODEL_DIR / "adversary_0.pth"
    if adversary_path.exists():
        actor = load_agent(adversary_path, obs_dim=16, action_dim=5, hidden_dim=128)
        if actor:
            agents_dict['adversary_0'] = actor
    
    logger.info(f"   Loaded {len(agents_dict)}/{len(env.agents)} agents")
    
    if not agents_dict:
        logger.error("❌ No agents loaded!")
        return
    
    # Evaluate
    logger.info(f"\n🎮 Running {NUM_EVAL_EPISODES} episodes...")
    
    all_total_rewards = []
    all_agent_rewards = {agent: [] for agent in env.agents}
    
    for ep in range(NUM_EVAL_EPISODES):
        total_reward, agent_rewards, steps = evaluate_episode(env, agents_dict)
        
        all_total_rewards.append(total_reward)
        for agent, reward in agent_rewards.items():
            all_agent_rewards[agent].append(reward)
        
        if (ep + 1) % 20 == 0:
            logger.info(f"   Progress: {ep + 1}/{NUM_EVAL_EPISODES}")
    
    # Results
    logger.info("\n" + "=" * 80)
    logger.info("📊 RESULTS")
    logger.info("=" * 80)
    
    avg_total = np.mean(all_total_rewards)
    std_total = np.std(all_total_rewards)
    
    logger.info(f"\n🎯 Total Reward:      {avg_total:.2f} ± {std_total:.2f}")
    logger.info(f"   Max:               {max(all_total_rewards):.2f}")
    logger.info(f"   Min:               {min(all_total_rewards):.2f}")
    
    logger.info(f"\n👥 Per-Agent Rewards:")
    for agent in env.agents:
        rewards = all_agent_rewards[agent]
        logger.info(f"   {agent:12s}: {np.mean(rewards):6.2f} ± {np.std(rewards):.2f}")
    
    success_rate = sum(1 for r in all_total_rewards if r > 0) / NUM_EVAL_EPISODES
    logger.info(f"\n✅ Success Rate:      {success_rate * 100:.1f}%")
    
    logger.info("=" * 80)
    
    env.close()

if __name__ == '__main__':
    main()
