"""
Evaluate trained MADDPG agents on simple_tag_v3
Environment: simple_tag_v3 with num_good=1, num_adversaries=1, num_obstacles=0
"""

import torch
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

from pettingzoo.mpe import simple_tag_v3

# ==================== MODEL DEFINITION ====================
class ActorNetwork(torch.nn.Module):
    """Actor network matching training structure"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# ==================== LOAD AGENTS ====================
def load_agent(checkpoint_path, agent_name):
    """Load a single agent from checkpoint"""
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract dimensions from network structure
    first_layer = checkpoint['actor']['network.0.weight']
    obs_dim = first_layer.shape[1]
    action_dim = checkpoint['actor']['network.4.weight'].shape[0]
    
    # Create and load actor
    actor = ActorNetwork(obs_dim, action_dim)
    actor.load_state_dict(checkpoint['actor'])
    actor.eval()
    
    logger.info(f"✅ Loaded {agent_name}: obs_dim={obs_dim}, action_dim={action_dim}")
    
    return {
        'actor': actor,
        'obs_dim': obs_dim,
        'action_dim': action_dim
    }

def load_all_agents(checkpoint_dir):
    """Load all trained agents"""
    checkpoint_path = Path(checkpoint_dir)
    
    agents = {}
    
    # Load good agent
    agents['agent_0'] = load_agent(checkpoint_path / "agent_0.pth", "agent_0")
    
    # Load adversary
    agents['adversary_0'] = load_agent(checkpoint_path / "adversary_0.pth", "adversary_0")
    
    return agents

# ==================== EVALUATION ====================
def evaluate_simple_tag(agents, num_episodes=10, render=False):
    """Evaluate agents in simple_tag_v3"""
    
    logger.info("=" * 80)
    logger.info("🎮 STARTING EVALUATION: simple_tag_v3")
    logger.info("=" * 80)
    
    # Create environment with exact parameters
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=1,
        num_obstacles=0,
        max_cycles=25,
        continuous_actions=True,
        render_mode='human' if render else None
    )
    
    episode_rewards = {agent: [] for agent in ['agent_0', 'adversary_0']}
    episode_lengths = []
    adversary_catches = []
    
    for episode in range(num_episodes):
        observations, infos = env.reset()
        
        episode_reward = {agent: 0 for agent in ['agent_0', 'adversary_0']}
        step = 0
        done = False
        catches = 0
        
        while not done:
            actions = {}
            
            # Get actions for all agents
            for agent_name in observations.keys():
                obs = torch.FloatTensor(observations[agent_name]).unsqueeze(0)
                
                with torch.no_grad():
                    action = agents[agent_name]['actor'](obs).squeeze(0).numpy()
                
                actions[agent_name] = action
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Accumulate rewards
            for agent_name in rewards.keys():
                episode_reward[agent_name] += rewards[agent_name]
            
            # Count catches (adversary gets positive reward when catching)
            if rewards['adversary_0'] > 0:
                catches += 1
            
            step += 1
            done = all(terminations.values()) or all(truncations.values())
        
        # Store episode results
        for agent in episode_reward.keys():
            episode_rewards[agent].append(episode_reward[agent])
        episode_lengths.append(step)
        adversary_catches.append(catches)
        
        logger.info(f"Episode {episode + 1}/{num_episodes}: "
                   f"agent_0={episode_reward['agent_0']:.2f}, "
                   f"adversary_0={episode_reward['adversary_0']:.2f}, "
                   f"catches={catches}, steps={step}")
    
    env.close()
    
    # Calculate statistics
    logger.info("\n" + "=" * 80)
    logger.info("📊 EVALUATION RESULTS")
    logger.info("=" * 80)
    
    for agent in episode_rewards.keys():
        mean_reward = np.mean(episode_rewards[agent])
        std_reward = np.std(episode_rewards[agent])
        logger.info(f"{agent}: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    mean_catches = np.mean(adversary_catches)
    mean_length = np.mean(episode_lengths)
    logger.info(f"\nMean Catches per Episode: {mean_catches:.1f}")
    logger.info(f"Mean Episode Length: {mean_length:.1f} steps")
    logger.info("=" * 80)
    
    return episode_rewards, episode_lengths, adversary_catches

# ==================== MAIN ====================
def main():
    # Configuration
    CHECKPOINT_DIR = "models/level1_simple/final"
    NUM_EPISODES = 10
    RENDER = False  # Set to True to visualize
    
    try:
        # Load trained agents
        logger.info("🔄 Loading trained agents...")
        agents = load_all_agents(CHECKPOINT_DIR)
        
        # Evaluate
        episode_rewards, episode_lengths, catches = evaluate_simple_tag(
            agents,
            num_episodes=NUM_EPISODES,
            render=RENDER
        )
        
        logger.info("\n✅ Evaluation completed successfully!")
        
        # Performance analysis
        agent_wins = sum(1 for r in episode_rewards['agent_0'] if r > episode_rewards['adversary_0'][episode_rewards['agent_0'].index(r)])
        logger.info(f"\n🏆 Agent escaped in {agent_wins}/{NUM_EPISODES} episodes")
        logger.info(f"🏆 Adversary caught in {NUM_EPISODES - agent_wins}/{NUM_EPISODES} episodes")
        
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
