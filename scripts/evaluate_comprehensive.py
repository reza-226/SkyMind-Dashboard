# scripts/evaluate_comprehensive.py

"""
Comprehensive Evaluation Script
=================================
مقایسه کامل همه روش‌ها برای فصل 4
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try different import patterns
try:
    from core.env.environment import UAVOffloadingEnv
except ImportError:
    try:
        from core.env.environment import Environment as UAVOffloadingEnv
    except ImportError:
        try:
            from core.env.environment import OffloadingEnvironment as UAVOffloadingEnv
        except ImportError:
            # Last resort - import module and find the class
            import core.env.environment as env_module
            
            # Find the main environment class
            for name in dir(env_module):
                obj = getattr(env_module, name)
                if isinstance(obj, type) and 'env' in name.lower():
                    UAVOffloadingEnv = obj
                    print(f"✅ Found environment class: {name}")
                    break
            else:
                raise ImportError("Could not find environment class")

from algorithms.baselines.simple_policies import RandomPolicy, GreedyLocalPolicy
from algorithms.baselines.dqn_agent import DQNAgent
from algorithms.baselines.ddpg_agent import DDPGAgent


class ComprehensiveEvaluator:
    """ارزیابی کامل همه روش‌ها"""
    
    def __init__(self, num_episodes=100):
        self.num_episodes = num_episodes
        self.results_dir = project_root / "results" / "comprehensive_evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment with error handling
        try:
            self.env = UAVOffloadingEnv(
                num_uavs=3,
                num_tasks=100,
                render_mode=None
            )
        except TypeError:
            # Try without render_mode
            try:
                self.env = UAVOffloadingEnv(
                    num_uavs=3,
                    num_tasks=100
                )
            except TypeError:
                # Try with minimal args
                self.env = UAVOffloadingEnv()
        
        print("=" * 70)
        print("🚀 Comprehensive Baseline Evaluation")
        print("=" * 70)
        print(f"📊 Environment initialized: {type(self.env).__name__}")
        print(f"📈 Episodes per method: {num_episodes}")
        print(f"💾 Results directory: {self.results_dir}")
        print("=" * 70 + "\n")
    
    def evaluate_policy(self, policy_name, agent, description):
        """ارزیابی یک policy"""
        print(f"\n{'=' * 70}")
        print(f"📍 Evaluating: {policy_name}")
        print(f"📝 Description: {description}")
        print(f"{'=' * 70}")
        
        episode_rewards = []
        episode_metrics = {
            'delay': [],
            'energy': [],
            'distance': [],
            'qos': []
        }
        
        for episode in tqdm(range(self.num_episodes), desc=f"  {policy_name}"):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            # Metrics accumulation
            ep_delay = []
            ep_energy = []
            ep_distance = []
            ep_qos = []
            
            while not done:
                # Select action
                if hasattr(agent, 'select_action'):
                    action = agent.select_action(state, evaluation=True)
                else:
                    action = agent.get_action(state)
                
                # Step environment
                try:
                    next_state, reward, done, info = self.env.step(action)
                except ValueError as e:
                    # Handle case where step returns 5 values (with truncated)
                    result = self.env.step(action)
                    if len(result) == 5:
                        next_state, reward, terminated, truncated, info = result
                        done = terminated or truncated
                    else:
                        raise e
                
                episode_reward += reward
                step_count += 1
                
                # Collect metrics
                if isinstance(info, dict):
                    ep_delay.append(info.get('delay', 0))
                    ep_energy.append(info.get('energy_consumption', 0))
                    ep_distance.append(info.get('distance', 0))
                    ep_qos.append(info.get('qos_satisfaction', 0))
                
                state = next_state
                
                if step_count > 500:  # Safety limit
                    break
            
            episode_rewards.append(episode_reward)
            
            # Store episode metrics
            if ep_delay:
                episode_metrics['delay'].append(np.mean(ep_delay))
                episode_metrics['energy'].append(np.mean(ep_energy))
                episode_metrics['distance'].append(np.mean(ep_distance))
                episode_metrics['qos'].append(np.mean(ep_qos))
        
        # Calculate statistics
        results = {
            'policy_name': policy_name,
            'description': description,
            'num_episodes': self.num_episodes,
            'rewards': {
                'mean': float(np.mean(episode_rewards)),
                'std': float(np.std(episode_rewards)),
                'min': float(np.min(episode_rewards)),
                'max': float(np.max(episode_rewards)),
                'all': [float(r) for r in episode_rewards]
            },
            'metrics': {
                'delay': {
                    'mean': float(np.mean(episode_metrics['delay'])) if episode_metrics['delay'] else 0,
                    'std': float(np.std(episode_metrics['delay'])) if episode_metrics['delay'] else 0
                },
                'energy': {
                    'mean': float(np.mean(episode_metrics['energy'])) if episode_metrics['energy'] else 0,
                    'std': float(np.std(episode_metrics['energy'])) if episode_metrics['energy'] else 0
                },
                'distance': {
                    'mean': float(np.mean(episode_metrics['distance'])) if episode_metrics['distance'] else 0,
                    'std': float(np.std(episode_metrics['distance'])) if episode_metrics['distance'] else 0
                },
                'qos': {
                    'mean': float(np.mean(episode_metrics['qos'])) if episode_metrics['qos'] else 0,
                    'std': float(np.std(episode_metrics['qos'])) if episode_metrics['qos'] else 0
                }
            }
        }
        
        # Print summary
        print(f"\n📊 Results for {policy_name}:")
        print(f"   Avg Reward: {results['rewards']['mean']:.4f} ± {results['rewards']['std']:.4f}")
        print(f"   Avg Delay: {results['metrics']['delay']['mean']:.4f} ± {results['metrics']['delay']['std']:.4f}")
        print(f"   Avg Energy: {results['metrics']['energy']['mean']:.4f} ± {results['metrics']['energy']['std']:.4f}")
        print(f"   Avg QoS: {results['metrics']['qos']['mean']:.4f} ± {results['metrics']['qos']['std']:.4f}")
        
        return results
    
    def run_evaluation(self):
        """اجرای ارزیابی کامل"""
        all_results = {}
        
        # 1. Random Policy
        print("\n" + "🎲" * 35)
        random_agent = RandomPolicy()
        all_results['random'] = self.evaluate_policy(
            'Random Policy',
            random_agent,
            'Baseline: Random action selection'
        )
        
        # 2. Greedy Local Policy
        print("\n" + "🎯" * 35)
        greedy_agent = GreedyLocalPolicy()
        all_results['greedy_local'] = self.evaluate_policy(
            'Greedy Local Policy',
            greedy_agent,
            'Baseline: Always local processing'
        )
        
        # 3. DQN Agent
        print("\n" + "🤖" * 35)
        dqn_path = project_root / "results" / "baselines" / "dqn" / "dqn_model.pt"
        if dqn_path.exists():
            dqn_agent = DQNAgent(state_dim=537, action_dim=5)
            dqn_agent.load(str(dqn_path))
            all_results['dqn'] = self.evaluate_policy(
                'DQN',
                dqn_agent,
                'Deep Q-Network with discrete offloading'
            )
        else:
            print(f"⚠️  DQN model not found - using untrained agent for comparison")
            dqn_agent = DQNAgent(state_dim=537, action_dim=5)
            all_results['dqn'] = self.evaluate_policy(
                'DQN (untrained)',
                dqn_agent,
                'Deep Q-Network (untrained baseline)'
            )
        
        # 4. DDPG Agent
        print("\n" + "🎯" * 35)
        ddpg_path = project_root / "results" / "baselines" / "ddpg" / "ddpg_model.pt"
        if ddpg_path.exists():
            ddpg_agent = DDPGAgent(state_dim=537, action_dim=6)
            ddpg_agent.load(str(ddpg_path))
            all_results['ddpg'] = self.evaluate_policy(
                'DDPG',
                ddpg_agent,
                'Deep Deterministic Policy Gradient'
            )
        else:
            print(f"⚠️  DDPG model not found - using untrained agent for comparison")
            ddpg_agent = DDPGAgent(state_dim=537, action_dim=6)
            all_results['ddpg'] = self.evaluate_policy(
                'DDPG (untrained)',
                ddpg_agent,
                'DDPG (untrained baseline)'
            )
        
        # Save results
        self._save_results(all_results)
        self._print_comparison_table(all_results)
        
        return all_results
    
    def _save_results(self, results):
        """ذخیره نتایج"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"comprehensive_results_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
    
    def _print_comparison_table(self, results):
        """چاپ جدول مقایسه"""
        print("\n" + "=" * 100)
        print("📊 COMPREHENSIVE COMPARISON TABLE")
        print("=" * 100)
        
        print(f"\n{'Method':<20} {'Avg Reward':<15} {'Delay':<15} {'Energy':<15} {'QoS':<15}")
        print("-" * 100)
        
        for method_key, data in results.items():
            name = data['policy_name']
            reward = f"{data['rewards']['mean']:.4f}"
            delay = f"{data['metrics']['delay']['mean']:.4f}"
            energy = f"{data['metrics']['energy']['mean']:.4f}"
            qos = f"{data['metrics']['qos']['mean']:.4f}"
            
            print(f"{name:<20} {reward:<15} {delay:<15} {energy:<15} {qos:<15}")
        
        print("=" * 100)


def main():
    evaluator = ComprehensiveEvaluator(num_episodes=100)
    results = evaluator.run_evaluation()
    
    print("\n✅ Comprehensive evaluation completed!")
    print(f"📂 Results saved in: {evaluator.results_dir}")


if __name__ == "__main__":
    main()
