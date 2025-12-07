"""
Comprehensive Evaluation Script for PettingZoo MPE MADDPG Agent
Fixed: Correct action space mapping [0, 1]
"""

import torch
import numpy as np
try:
    from pettingzoo.mpe import simple_tag_v3
except:
    from mpe2 import simple_tag_v3
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import ActorNetwork
try:
    from models.actor_critic.actor_network import ActorNetwork
    print("✅ Imported ActorNetwork from models.actor_critic")
except ImportError:
    try:
        from algorithms.baselines.iddpg.actor_network import ActorNetwork
        print("✅ Imported ActorNetwork from algorithms.baselines.iddpg")
    except ImportError:
        raise ImportError("❌ Could not import ActorNetwork from either location!")


class MPEEvaluator:
    """Evaluator for PettingZoo MPE simple_tag environment"""
    
    def __init__(self, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        # Environment setup
        self.env = simple_tag_v3.parallel_env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=25,
            continuous_actions=True
        )
        
        # Get observation dimensions
        self.adversary_obs_dim = 16
        self.agent_obs_dim = 14
        
        print(f"\n🎮 Environment Setup:")
        print(f"   Environment: simple_tag_v3")
        print(f"   Adversary obs dim: {self.adversary_obs_dim}")
        print(f"   Agent obs dim: {self.agent_obs_dim}")
        print(f"   Action space: Box(0.0, 1.0, (5,), float32)")
        print(f"   Device: {self.device}")
        
        # Load models
        self.actors = self._load_models()
        
    def _load_models(self):
        """Load actor networks for all agents"""
        actors = {}
        
        # Reset to get agent names
        self.env.reset()
        agent_names = self.env.possible_agents
        
        print(f"\n📥 Loading Models:")
        print(f"   Checkpoint: {self.checkpoint_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"❌ Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Check checkpoint structure
        print(f"   Checkpoint keys: {list(checkpoint.keys())}")
        
        # Handle 'actors' key - can be list or dict
        if 'actors' in checkpoint:
            actor_states = checkpoint['actors']
            
            # Check if it's a list or dict
            if isinstance(actor_states, list):
                print(f"   Found 'actors' list with {len(actor_states)} items")
                
                # Convert list to dict by matching with agent names
                if len(actor_states) == len(agent_names):
                    actor_states_dict = {agent_names[i]: actor_states[i] 
                                        for i in range(len(agent_names))}
                    actor_states = actor_states_dict
                    print(f"   Converted to dict with keys: {list(actor_states.keys())}")
                else:
                    print(f"   ⚠️ Mismatch: {len(actor_states)} actors vs {len(agent_names)} agents")
                    actor_states_dict = {}
                    idx = 0
                    # First adversaries
                    for agent in sorted(agent_names):
                        if 'adversary' in agent and idx < len(actor_states):
                            actor_states_dict[agent] = actor_states[idx]
                            idx += 1
                    # Then good agent
                    for agent in sorted(agent_names):
                        if 'agent' in agent and idx < len(actor_states):
                            actor_states_dict[agent] = actor_states[idx]
                            idx += 1
                    actor_states = actor_states_dict
                    print(f"   Matched by order: {list(actor_states.keys())}")
            
            elif isinstance(actor_states, dict):
                print(f"   Found 'actors' dict with keys: {list(actor_states.keys())}")
            else:
                raise ValueError(f"Unknown actors format: {type(actor_states)}")
        else:
            actor_states = checkpoint
        
        # Load weights for each agent
        for agent_name in agent_names:
            # Determine observation dimension
            if 'adversary' in agent_name:
                obs_dim = self.adversary_obs_dim
            else:
                obs_dim = self.agent_obs_dim
            
            # Create actor network
            actor = ActorNetwork(
                state_dim=obs_dim,
                offload_dim=5,
                continuous_dim=6,
                hidden_dim=512
            ).to(self.device)
            
            # Try to load weights
            loaded = False
            
            if agent_name in actor_states:
                state_dict = actor_states[agent_name]
                if isinstance(state_dict, dict):
                    model_dict = actor.state_dict()
                    
                    # Filter compatible weights
                    filtered_dict = {k: v for k, v in state_dict.items() 
                                    if k in model_dict and v.shape == model_dict[k].shape}
                    
                    if filtered_dict:
                        model_dict.update(filtered_dict)
                        actor.load_state_dict(model_dict)
                        print(f"   ✅ {agent_name}: loaded {len(filtered_dict)}/{len(state_dict)} layers")
                        loaded = True
            
            if not loaded:
                print(f"   ⚠️  {agent_name}: using random initialization")
            
            actor.eval()
            actors[agent_name] = actor
        
        return actors
    
    def select_action(self, agent_name, observation, deterministic=True):
        """
        Select action for a single agent
        MPE expects: Box(0.0, 1.0, (5,), float32)
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            offload_logits, continuous_output = self.actors[agent_name](obs_tensor)
            
            # Get discrete action (0-4)
            if deterministic:
                discrete_action = torch.argmax(offload_logits, dim=1).item()
            else:
                probs = torch.softmax(offload_logits, dim=1)
                discrete_action = torch.multinomial(probs, 1).item()
            
            # Create 5D action vector for MPE - ALL VALUES IN [0, 1]
            action = np.zeros(5, dtype=np.float32)
            
            # Map discrete action to movement directions
            # Movement dimensions use values in [0, 1]
            if discrete_action == 0:  # No movement (center)
                action[:2] = [0.5, 0.5]
            elif discrete_action == 1:  # Right
                action[:2] = [1.0, 0.5]
            elif discrete_action == 2:  # Left
                action[:2] = [0.0, 0.5]
            elif discrete_action == 3:  # Up
                action[:2] = [0.5, 1.0]
            elif discrete_action == 4:  # Down
                action[:2] = [0.5, 0.0]
            
            # Communication/other actions (also need to be in [0, 1])
            # Convert continuous_output from [-1, 1] to [0, 1]
            comm_actions = (continuous_output[0, :3].cpu().numpy() + 1.0) / 2.0
            action[2:] = np.clip(comm_actions, 0.0, 1.0)
            
        return action
    
    def evaluate_episode(self, render=False, deterministic=True):
        """Run one evaluation episode"""
        reset_result = self.env.reset()
        
        if isinstance(reset_result, tuple):
            observations, _ = reset_result
        else:
            observations = reset_result
        
        episode_rewards = {agent: 0.0 for agent in self.env.possible_agents}
        steps = 0
        
        while self.env.agents:
            actions = {}
            
            for agent_name in self.env.agents:
                obs = observations[agent_name]
                action = self.select_action(agent_name, obs, deterministic=deterministic)
                actions[agent_name] = action
            
            step_result = self.env.step(actions)
            observations = step_result[0]
            rewards = step_result[1]
            
            for agent_name, reward in rewards.items():
                episode_rewards[agent_name] += reward
            
            steps += 1
            
            if render:
                self.env.render()
        
        return episode_rewards, steps
    
    def run_evaluation(self, num_episodes=100, save_results=True):
        """Run comprehensive evaluation"""
        print(f"\n{'='*70}")
        print(f"🎯 Starting Evaluation - {num_episodes} Episodes")
        print(f"{'='*70}\n")
        
        all_results = []
        
        for episode in range(num_episodes):
            episode_rewards, steps = self.evaluate_episode()
            
            total_reward = sum(episode_rewards.values())
            avg_reward = total_reward / len(episode_rewards)
            agent_reward = episode_rewards.get('agent_0', 0.0)
            adversary_avg = np.mean([episode_rewards.get(f'adversary_{i}', 0.0) 
                                     for i in range(3)])
            
            # Success criteria: agent avoids being caught
            success = agent_reward > -10.0
            
            result = {
                'episode': episode,
                'total_reward': float(total_reward),
                'avg_reward': float(avg_reward),
                'agent_reward': float(agent_reward),
                'adversary_avg': float(adversary_avg),
                'success': bool(success),
                'steps': int(steps),
                'rewards_per_agent': {k: float(v) for k, v in episode_rewards.items()}
            }
            all_results.append(result)
            
            if (episode + 1) % 10 == 0:
                recent = all_results[-10:]
                recent_agent = [r['agent_reward'] for r in recent]
                recent_adv = [r['adversary_avg'] for r in recent]
                recent_success = sum([r['success'] for r in recent])
                
                print(f"Episode {episode+1:3d} | "
                      f"Agent: {agent_reward:7.3f} | "
                      f"Adversaries: {adversary_avg:7.3f} | "
                      f"Success: {success} | "
                      f"Avg(10): Agent={np.mean(recent_agent):7.3f}, "
                      f"Adv={np.mean(recent_adv):7.3f}, "
                      f"Success={recent_success}/10")
        
        stats = self._compute_statistics(all_results)
        self._print_results(stats)
        
        if save_results:
            self._save_results(all_results, stats)
        
        return all_results, stats
    
    def _compute_statistics(self, results):
        """Compute evaluation statistics"""
        agent_rewards = [r['agent_reward'] for r in results]
        adversary_rewards = [r['adversary_avg'] for r in results]
        total_rewards = [r['total_reward'] for r in results]
        steps = [r['steps'] for r in results]
        successes = [r['success'] for r in results]
        
        return {
            'num_episodes': len(results),
            'success_rate': float(np.mean(successes)),
            'agent_reward': {
                'mean': float(np.mean(agent_rewards)),
                'std': float(np.std(agent_rewards)),
                'min': float(np.min(agent_rewards)),
                'max': float(np.max(agent_rewards)),
                'median': float(np.median(agent_rewards))
            },
            'adversary_reward': {
                'mean': float(np.mean(adversary_rewards)),
                'std': float(np.std(adversary_rewards)),
                'min': float(np.min(adversary_rewards)),
                'max': float(np.max(adversary_rewards))
            },
            'total_reward': {
                'mean': float(np.mean(total_rewards)),
                'std': float(np.std(total_rewards))
            },
            'steps': {
                'mean': float(np.mean(steps)),
                'std': float(np.std(steps)),
                'min': int(np.min(steps)),
                'max': int(np.max(steps))
            }
        }
    
    def _print_results(self, stats):
        """Print evaluation results"""
        print(f"\n{'='*70}")
        print("📊 Evaluation Results Summary")
        print(f"{'='*70}\n")
        
        print(f"Total Episodes: {stats['num_episodes']}")
        print(f"Success Rate:   {stats['success_rate']*100:.1f}% ({int(stats['success_rate']*stats['num_episodes'])}/{stats['num_episodes']})")
        
        print(f"\n🎯 Good Agent Performance:")
        print(f"   Mean Reward:   {stats['agent_reward']['mean']:7.3f} ± {stats['agent_reward']['std']:.3f}")
        print(f"   Median Reward: {stats['agent_reward']['median']:7.3f}")
        print(f"   Range:         [{stats['agent_reward']['min']:7.3f}, {stats['agent_reward']['max']:7.3f}]")
        
        print(f"\n👾 Adversaries Performance:")
        print(f"   Mean Reward:   {stats['adversary_reward']['mean']:7.3f} ± {stats['adversary_reward']['std']:.3f}")
        print(f"   Range:         [{stats['adversary_reward']['min']:7.3f}, {stats['adversary_reward']['max']:7.3f}]")
        
        print(f"\n📈 Episode Statistics:")
        print(f"   Avg Steps:     {stats['steps']['mean']:.1f} ± {stats['steps']['std']:.1f}")
        print(f"   Step Range:    [{stats['steps']['min']}, {stats['steps']['max']}]")
        print(f"   Total Reward:  {stats['total_reward']['mean']:.3f} ± {stats['total_reward']['std']:.3f}")
        
        # Performance interpretation
        agent_mean = stats['agent_reward']['mean']
        adv_mean = stats['adversary_reward']['mean']
        success_rate = stats['success_rate']
        
        print(f"\n💡 Performance Interpretation:")
        if success_rate >= 0.7:
            print(f"   ✅ Good performance: Agent escaping successfully {success_rate*100:.1f}% of time")
        elif success_rate >= 0.4:
            print(f"   ⚠️  Moderate performance: Agent escaping {success_rate*100:.1f}% of time")
        else:
            print(f"   ❌ Poor performance: Agent caught frequently (success {success_rate*100:.1f}%)")
        
        if agent_mean > adv_mean:
            print(f"   ✅ Agent rewards higher than adversaries (Δ = {agent_mean - adv_mean:+.3f})")
        else:
            print(f"   ⚠️  Adversaries rewards higher (Δ = {agent_mean - adv_mean:+.3f})")
    
    def _save_results(self, results, stats):
        """Save evaluation results to file"""
        output_dir = Path('evaluation_results')
        output_dir.mkdir(exist_ok=True)
        
        checkpoint_name = self.checkpoint_path.stem
        output_file = output_dir / f'eval_{checkpoint_name}.json'
        
        output_data = {
            'checkpoint': str(self.checkpoint_path),
            'environment': 'simple_tag_v3',
            'device': str(self.device),
            'statistics': stats,
            'episode_results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    """Main evaluation function"""
    
    checkpoints = {
        'best': 'checkpoints/maddpg/best_model.pt',
        'ep100': 'checkpoints/maddpg/checkpoint_episode_100.pt',
        'ep200': 'checkpoints/maddpg/checkpoint_episode_200.pt',
        'ep300': 'checkpoints/maddpg/checkpoint_episode_300.pt',
        'ep400': 'checkpoints/maddpg/checkpoint_episode_400.pt',
        'ep500': 'checkpoints/maddpg/checkpoint_episode_500.pt'
    }
    
    print("\n📋 Available Checkpoints:")
    for name, path in checkpoints.items():
        exists = "✅" if Path(path).exists() else "❌"
        print(f"   {exists} {name:8s} → {path}")
    
    checkpoint_to_use = 'best'
    checkpoint_path = checkpoints[checkpoint_to_use]
    
    if not Path(checkpoint_path).exists():
        print(f"\n❌ Error: Selected checkpoint '{checkpoint_to_use}' not found!")
        return
    
    print(f"\n🚀 Evaluating Checkpoint: {checkpoint_to_use}")
    
    evaluator = MPEEvaluator(
        checkpoint_path=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    results, stats = evaluator.run_evaluation(
        num_episodes=100,
        save_results=True
    )
    
    print(f"\n{'='*70}")
    print("✅ Evaluation Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
