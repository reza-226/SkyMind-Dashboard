"""
FINAL Evaluation Script - Exact Match with Trained Models
Works with: obs_dim=14 (agent) and obs_dim=16 (adversary)
"""

import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ActorNetwork(torch.nn.Module):
    """Actor Network matching checkpoint structure"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Sigmoid()  # Actions in [0,1]
        )
    
    def forward(self, obs):
        return self.network(obs)

def load_agent(checkpoint_path: Path, obs_dim: int, action_dim: int):
    """Load trained agent from checkpoint"""
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'actor' not in checkpoint:
            raise ValueError(f"No 'actor' key in checkpoint: {checkpoint_path}")
        
        actor = ActorNetwork(obs_dim, action_dim, hidden_dim=128)
        actor.load_state_dict(checkpoint['actor'])
        actor.eval()
        
        logger.info(f"✅ Loaded: {checkpoint_path.name} (obs_dim={obs_dim})")
        
        return actor
        
    except Exception as e:
        logger.error(f"❌ Failed to load {checkpoint_path}: {e}")
        raise

def create_environment(env_type: str = "spread"):
    """
    Create environment matching trained models
    
    env_type:
        "spread" → simple_spread_v3 with N=2 (obs_dim=14)
        "adversary" → simple_adversary_v3 custom config (obs_dim=16)
    """
    
    if env_type == "spread":
        from pettingzoo.mpe import simple_spread_v3
        env = simple_spread_v3.parallel_env(N=2, continuous_actions=True)
        logger.info("✅ Created: simple_spread_v3 (N=2)")
        
    elif env_type == "adversary":
        from pettingzoo.mpe import simple_adversary_v3
        # Custom configuration: 2 good agents, 1 adversary, 2 landmarks
        env = simple_adversary_v3.parallel_env(
            N=2,  # Number of good agents
            continuous_actions=True
        )
        logger.info("✅ Created: simple_adversary_v3 (N=2 good agents)")
    
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
    
    return env

def verify_observations(env, expected_dims: Dict[str, int]):
    """Verify environment observation dimensions match model expectations"""
    
    observations, _ = env.reset()
    
    logger.info("\n🔍 Verifying observation dimensions:")
    
    all_match = True
    for agent_name, obs in observations.items():
        obs_dim = obs.shape[0]
        expected = expected_dims.get(agent_name, None)
        
        status = "✅" if obs_dim == expected else "❌"
        logger.info(f"   {status} {agent_name}: obs_dim={obs_dim} (expected={expected})")
        
        if obs_dim != expected:
            all_match = False
    
    if not all_match:
        raise ValueError("❌ Observation dimensions don't match model expectations!")
    
    logger.info("✅ All observation dimensions match!\n")
    return True

def evaluate_episode(env, agents: Dict[str, ActorNetwork], render: bool = False):
    """Run one evaluation episode"""
    
    observations, _ = env.reset()
    episode_reward = {agent: 0.0 for agent in env.agents}
    done = False
    step = 0
    
    while not done:
        actions = {}
        
        for agent_name in env.agents:
            obs_tensor = torch.FloatTensor(observations[agent_name]).unsqueeze(0)
            
            with torch.no_grad():
                action = agents[agent_name](obs_tensor).squeeze(0).numpy()
            
            actions[agent_name] = action
        
        observations, rewards, terminations, truncations, _ = env.step(actions)
        
        for agent_name in env.agents:
            episode_reward[agent_name] += rewards[agent_name]
        
        done = any(terminations.values()) or any(truncations.values())
        step += 1
        
        if render:
            env.render()
    
    return episode_reward, step

def main():
    logger.info("=" * 80)
    logger.info("🎯 FINAL EVALUATION - Exact Environment Match")
    logger.info("=" * 80)
    
    # Configuration
    checkpoint_dir = Path("models/level3_complex/final")
    num_episodes = 10
    env_type = "spread"  # Change to "adversary" to test adversary env
    
    # Expected dimensions (from checkpoint analysis)
    expected_dims = {
        'agent_0': 14,
        'agent_1': 14,  # In spread mode
        'adversary_0': 16  # In adversary mode
    }
    
    logger.info(f"\n📁 Loading models from: {checkpoint_dir}")
    logger.info(f"🎮 Environment type: {env_type}")
    logger.info(f"📊 Episodes to evaluate: {num_episodes}\n")
    
    # Create environment
    env = create_environment(env_type)
    
    # Verify dimensions match
    if env_type == "spread":
        verify_observations(env, {'agent_0': 14, 'agent_1': 14})
        
        # Load agents
        agents = {
            'agent_0': load_agent(checkpoint_dir / "agent_0.pth", obs_dim=14, action_dim=5),
            'agent_1': load_agent(checkpoint_dir / "agent_0.pth", obs_dim=14, action_dim=5)  # Share weights
        }
        
    elif env_type == "adversary":
        verify_observations(env, {'adversary_0': 16, 'agent_0': 10, 'agent_1': 10})
        
        logger.warning("⚠️  Adversary mode: good agents (obs=10) don't have matching checkpoints!")
        logger.warning("    Only adversary (obs=16) can be evaluated.")
        env.close()
        return
    
    # Evaluate
    logger.info(f"🚀 Starting evaluation ({num_episodes} episodes)...\n")
    
    all_rewards = []
    all_steps = []
    
    for ep in range(1, num_episodes + 1):
        episode_reward, steps = evaluate_episode(env, agents, render=False)
        
        total_reward = sum(episode_reward.values())
        all_rewards.append(total_reward)
        all_steps.append(steps)
        
        logger.info(f"Episode {ep:3d} | Reward: {total_reward:8.2f} | Steps: {steps:3d}")
    
    env.close()
    
    # Statistics
    logger.info("\n" + "=" * 80)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Mean Reward:   {np.mean(all_rewards):8.2f} ± {np.std(all_rewards):6.2f}")
    logger.info(f"Max Reward:    {np.max(all_rewards):8.2f}")
    logger.info(f"Min Reward:    {np.min(all_rewards):8.2f}")
    logger.info(f"Mean Steps:    {np.mean(all_steps):8.1f}")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
