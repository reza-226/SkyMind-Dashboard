"""
Evaluate all levels with shared agent models
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

# ==================== ENVIRONMENT CONFIGS ====================
ENV_CONFIGS = {
    'level1': {
        'num_good': 1,
        'num_adversaries': 1,
        'num_obstacles': 0,
        'expected_dims': {
            'agent': 6,
            'adversary': 8
        }
    },
    'level2': {
        'num_good': 2,
        'num_adversaries': 1,
        'num_obstacles': 0,
        'expected_dims': {
            'agent': 10,
            'adversary': 12
        }
    },
    'level3': {
        'num_good': 2,
        'num_adversaries': 1,
        'num_obstacles': 2,
        'expected_dims': {
            'agent': 14,
            'adversary': 16
        }
    }
}

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
    
    logger.info(f"  ✅ Loaded {agent_name}: obs_dim={obs_dim}, action_dim={action_dim}")
    
    return {
        'actor': actor,
        'obs_dim': obs_dim,
        'action_dim': action_dim
    }

def load_all_agents(checkpoint_dir, level_config):
    """Load all trained agents for a level - use shared models for similar agents"""
    checkpoint_path = Path(checkpoint_dir)
    
    agents = {}
    
    # Load agent model (shared by all good agents)
    agent_file = checkpoint_path / "agent_0.pth"
    if agent_file.exists():
        agent_model = load_agent(agent_file, "agent_0")
        agents['agent'] = agent_model  # Shared model for all agents
    
    # Load adversary model (shared by all adversaries)
    adversary_file = checkpoint_path / "adversary_0.pth"
    if adversary_file.exists():
        adversary_model = load_agent(adversary_file, "adversary_0")
        agents['adversary'] = adversary_model  # Shared model for all adversaries
    
    return agents

# ==================== VERIFY ENVIRONMENT ====================
def verify_environment(level_config):
    """Verify environment dimensions match expected"""
    
    env = simple_tag_v3.parallel_env(
        num_good=level_config['num_good'],
        num_adversaries=level_config['num_adversaries'],
        num_obstacles=level_config['num_obstacles'],
        max_cycles=25,
        continuous_actions=True
    )
    
    observations, _ = env.reset()
    
    logger.info("  🔍 Verifying environment dimensions:")
    all_match = True
    
    for agent_name, obs in observations.items():
        actual_dim = len(obs)
        
        # Determine expected dimension based on agent type
        if 'agent' in agent_name:
            expected_dim = level_config['expected_dims']['agent']
            agent_type = 'agent'
        else:
            expected_dim = level_config['expected_dims']['adversary']
            agent_type = 'adversary'
        
        match = "✅" if actual_dim == expected_dim else "❌"
        logger.info(f"    {match} {agent_name} ({agent_type}): expected={expected_dim}, actual={actual_dim}")
        
        if actual_dim != expected_dim:
            all_match = False
    
    env.close()
    return all_match

# ==================== EVALUATION ====================
def evaluate_level(agents, level_config, level_name, num_episodes=10):
    """Evaluate agents for a specific level"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🎮 EVALUATING {level_name.upper()}")
    logger.info(f"{'='*80}")
    
    # Create environment
    env = simple_tag_v3.parallel_env(
        num_good=level_config['num_good'],
        num_adversaries=level_config['num_adversaries'],
        num_obstacles=level_config['num_obstacles'],
        max_cycles=25,
        continuous_actions=True
    )
    
    # Get all agent names from environment
    observations, _ = env.reset()
    agent_names = list(observations.keys())
    
    episode_rewards = {agent: [] for agent in agent_names}
    episode_lengths = []
    adversary_catches = []
    agent_escapes = []
    
    for episode in range(num_episodes):
        observations, infos = env.reset()
        
        episode_reward = {agent: 0 for agent in agent_names}
        step = 0
        done = False
        catches = 0
        
        while not done:
            actions = {}
            
            # Get actions for all agents
            for agent_name in observations.keys():
                obs = torch.FloatTensor(observations[agent_name]).unsqueeze(0)
                
                # Use shared model based on agent type
                if 'agent' in agent_name:
                    model = agents['agent']
                else:
                    model = agents['adversary']
                
                with torch.no_grad():
                    action = model['actor'](obs).squeeze(0).numpy()
                
                actions[agent_name] = action
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Accumulate rewards
            for agent_name in rewards.keys():
                episode_reward[agent_name] += rewards[agent_name]
            
            # Count catches (adversary gets positive reward when catching)
            adversary_reward = sum(rewards[name] for name in agent_names if 'adversary' in name)
            if adversary_reward > 0:
                catches += 1
            
            step += 1
            done = all(terminations.values()) or all(truncations.values())
        
        # Store episode results
        for agent in episode_reward.keys():
            episode_rewards[agent].append(episode_reward[agent])
        episode_lengths.append(step)
        adversary_catches.append(catches)
        
        # Count escapes (agent gets positive final reward)
        agent_final_reward = sum(episode_reward[name] for name in agent_names if 'agent' in name)
        agent_escapes.append(1 if agent_final_reward > -10 else 0)
        
        logger.info(f"  Episode {episode + 1}/{num_episodes}: " + 
                   ", ".join([f"{name}={episode_reward[name]:.2f}" for name in sorted(agent_names)]) +
                   f", catches={catches}, steps={step}")
    
    env.close()
    
    # Calculate statistics
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 {level_name.upper()} RESULTS")
    logger.info(f"{'='*80}")
    
    for agent in sorted(episode_rewards.keys()):
        mean_reward = np.mean(episode_rewards[agent])
        std_reward = np.std(episode_rewards[agent])
        logger.info(f"  {agent}: Mean Reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    mean_catches = np.mean(adversary_catches)
    mean_escapes = np.mean(agent_escapes)
    mean_length = np.mean(episode_lengths)
    
    logger.info(f"\n  📈 Performance Metrics:")
    logger.info(f"    Mean Catches per Episode: {mean_catches:.1f}")
    logger.info(f"    Mean Escapes per Episode: {mean_escapes:.1f}")
    logger.info(f"    Mean Episode Length: {mean_length:.1f} steps")
    logger.info(f"    Success Rate (Escapes): {mean_escapes*10:.0f}%")
    logger.info(f"{'='*80}\n")
    
    return {
        'rewards': episode_rewards,
        'catches': adversary_catches,
        'escapes': agent_escapes,
        'lengths': episode_lengths
    }

# ==================== MAIN ====================
def main():
    """Evaluate all levels"""
    
    results = {}
    
    for level_name, level_config in ENV_CONFIGS.items():
        logger.info(f"\n{'#'*80}")
        logger.info(f"# {level_name.upper()}")
        logger.info(f"{'#'*80}")
        
        # Verify environment
        logger.info(f"\n🔧 Setting up environment...")
        logger.info(f"  num_good={level_config['num_good']}")
        logger.info(f"  num_adversaries={level_config['num_adversaries']}")
        logger.info(f"  num_obstacles={level_config['num_obstacles']}")
        
        if not verify_environment(level_config):
            logger.warning(f"⚠️  Dimension mismatch detected for {level_name}!")
            logger.warning(f"⚠️  Skipping evaluation...")
            continue
        
        # Load agents
        checkpoint_dir = f"models/{level_name}_simple/final" if level_name == 'level1' else \
                        f"models/{level_name}_medium/final" if level_name == 'level2' else \
                        f"models/{level_name}_complex/final"
        
        logger.info(f"\n🔄 Loading models from: {checkpoint_dir}")
        
        try:
            agents = load_all_agents(checkpoint_dir, level_config)
            
            # Evaluate
            results[level_name] = evaluate_level(agents, level_config, level_name, num_episodes=10)
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate {level_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("🏆 FINAL SUMMARY")
    logger.info(f"{'='*80}")
    
    for level_name, result in results.items():
        mean_escapes = np.mean(result['escapes'])
        logger.info(f"  {level_name.upper()}: Success Rate = {mean_escapes*100:.0f}%")
    
    logger.info(f"{'='*80}\n")

if __name__ == '__main__':
    main()
