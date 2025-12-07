"""
Comprehensive Evaluation Script - Single Agent Fix
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.actor_critic.actor_network import ActorNetwork
from environment.uav_env import UAVEnvironment

print(f"✅ ActorNetwork imported successfully")


class ComprehensiveEvaluator:
    """Comprehensive evaluation of trained MADDPG agents"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load checkpoint
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract configuration
        self.config = checkpoint.get('config', {})
        print(f"✅ Checkpoint config: {self.config}")
        
        # Get dimensions
        self.num_agents = len(checkpoint['actors'])
        self.state_dim = self.config.get('state_dim', 537)
        self.action_dim = self.config.get('action_dim', 5)
        
        print(f"🤖 Number of agents: {self.num_agents}")
        print(f"📊 State dim: {self.state_dim}, Action dim: {self.action_dim}")
        
        # Initialize actors
        self.actors = []
        for i, actor_state in enumerate(checkpoint['actors']):
            actor = ActorNetwork(
                state_dim=self.state_dim,
                offload_dim=self.action_dim,
                continuous_dim=6,
                hidden_dim=512
            ).to(self.device)
            
            # Partial loading
            self._load_partial_weights(actor, actor_state, agent_id=i)
            actor.eval()
            self.actors.append(actor)
        
        # Storage for results
        self.results = {
            'checkpoint': str(checkpoint_path),
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'scenarios': []
        }
        
        print("✅ Evaluator initialized!\n")
    
    def _load_partial_weights(self, actor, state_dict, agent_id):
        """Load only compatible weights from checkpoint"""
        model_state = actor.state_dict()
        filtered_state = {}
        
        print(f"\n  🔍 Loading weights for actor_{agent_id}:")
        
        for key, value in state_dict.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    filtered_state[key] = value
                    print(f"    ✅ {key}: {value.shape}")
                else:
                    print(f"    ⚠️  SKIP {key}: checkpoint {value.shape} vs model {model_state[key].shape}")
            else:
                print(f"    ⚠️  SKIP {key}: not in current model")
        
        actor.load_state_dict(filtered_state, strict=False)
        print(f"    ✅ Loaded {len(filtered_state)}/{len(state_dict)} weights")
        
        missing = set(model_state.keys()) - set(filtered_state.keys())
        if missing:
            print(f"    🔧 Randomly initialized: {missing}")
    
    def select_action(self, observation, agent_id=0):
        """
        ✅ Single agent action selection
        Returns: np.ndarray with shape (11,)
        """
        with torch.no_grad():
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # Get action from specified actor
            offload_logits, cont = self.actors[agent_id](obs_tensor)
            
            # Concatenate: [5 offload + 6 continuous] = 11 dims
            action = torch.cat([offload_logits, cont], dim=-1)
            
            # Return as 1D array
            return action.cpu().numpy()[0]  # Shape: (11,)
    
    def evaluate_scenario(self, scenario_name, num_tasks, task_complexity, max_steps, n_episodes):
        """Evaluate agents in a specific scenario"""
        print(f"\n🎯 Evaluating: {scenario_name}")
        print(f"   Tasks: {num_tasks}, Complexity: {task_complexity}, Steps: {max_steps}")
        
        env = UAVEnvironment(
            num_tasks=num_tasks,
            task_complexity=task_complexity,
            max_steps=max_steps
        )
        
        episode_rewards = []
        completion_rates = []
        energy_used = []
        collisions = []
        
        for ep in tqdm(range(n_episodes), desc=f"  {scenario_name}"):
            # Reset environment
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                observation = reset_result[0]
            else:
                observation = reset_result
            
            done = False
            total_reward = 0
            step = 0
            
            while not done and step < max_steps:
                # Select action for single agent
                action = self.select_action(observation, agent_id=0)
                
                # Step environment
                step_result = env.step(action)
                
                # Handle different return formats
                if len(step_result) == 5:
                    observation, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    observation, reward, done, info = step_result
                else:
                    observation, reward, done = step_result[:3]
                    info = {}
                
                total_reward += reward
                step += 1
            
            # Extract final info
            episode_rewards.append(total_reward)
            completion_rates.append(info.get('completion_rate', 0))
            energy_used.append(info.get('total_energy', 0))
            collisions.append(info.get('collisions', 0))
        
        env.close()
        
        scenario_result = {
            'name': scenario_name,
            'config': {
                'num_tasks': num_tasks,
                'task_complexity': task_complexity,
                'max_steps': max_steps
            },
            'metrics': {
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'min_reward': float(np.min(episode_rewards)),
                'max_reward': float(np.max(episode_rewards)),
                'mean_completion': float(np.mean(completion_rates)),
                'mean_energy': float(np.mean(energy_used)),
                'total_collisions': int(np.sum(collisions))
            },
            'raw_data': {
                'rewards': [float(r) for r in episode_rewards],
                'completions': [float(c) for c in completion_rates],
                'energy': [float(e) for e in energy_used]
            }
        }
        
        self.results['scenarios'].append(scenario_result)
        
        print(f"   ✅ Mean Reward: {scenario_result['metrics']['mean_reward']:.2f} ± {scenario_result['metrics']['std_reward']:.2f}")
        print(f"   ✅ Completion: {scenario_result['metrics']['mean_completion']:.2%}")
        print(f"   ✅ Collisions: {scenario_result['metrics']['total_collisions']}")
        
        return scenario_result
    
    def run_full_evaluation(self, n_episodes=50):
        """Run evaluation across multiple scenarios"""
        print("\n" + "="*60)
        print("🚀 COMPREHENSIVE EVALUATION STARTED")
        print("="*60)
        
        scenarios = [
            ("Easy Tasks", 50, 'easy', 200),
            ("Medium Tasks", 50, 'medium', 200),
            ("Hard Tasks", 50, 'hard', 200),
            ("Mixed Tasks", 50, 'mixed', 200),
            ("Few Tasks", 30, 'mixed', 200),
            ("Normal Tasks", 100, 'mixed', 500),
            ("Many Tasks", 200, 'mixed', 1000),
            ("Short Horizon", 50, 'mixed', 100),
            ("Long Horizon", 50, 'mixed', 500),
        ]
        
        for scenario_name, num_tasks, complexity, max_steps in scenarios:
            self.evaluate_scenario(
                scenario_name, num_tasks, complexity, max_steps, n_episodes
            )
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETED!")
        print("="*60)
        
        self.save_results()
        self.generate_report()
    
    def save_results(self):
        """Save evaluation results to JSON"""
        output_dir = Path('evaluation/results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'comprehensive_eval_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def generate_report(self):
        """Generate summary report"""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        for scenario in self.results['scenarios']:
            print(f"\n{scenario['name']}:")
            print(f"  Mean Reward: {scenario['metrics']['mean_reward']:.2f} ± {scenario['metrics']['std_reward']:.2f}")
            print(f"  Completion: {scenario['metrics']['mean_completion']:.2%}")
            print(f"  Collisions: {scenario['metrics']['total_collisions']}")
        
        self.plot_comparison()
    
    def plot_comparison(self):
        """Create comparison plots"""
        output_dir = Path('evaluation/plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scenario_names = [s['name'] for s in self.results['scenarios']]
        mean_rewards = [s['metrics']['mean_reward'] for s in self.results['scenarios']]
        std_rewards = [s['metrics']['std_reward'] for s in self.results['scenarios']]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(scenario_names))
        ax.bar(x, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Performance Across Different Scenarios')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = output_dir / f'scenario_comparison_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n📈 Plot saved to: {plot_file}")
        
        plt.close()


def main():
    """Main evaluation script"""
    checkpoint_path = 'checkpoints/maddpg/best_model.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    evaluator = ComprehensiveEvaluator(
        checkpoint_path=checkpoint_path,
        device='cuda'
    )
    
    evaluator.run_full_evaluation(n_episodes=50)
    
    print("\n🎉 Evaluation complete!")


if __name__ == '__main__':
    main()
