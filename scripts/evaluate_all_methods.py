# scripts/evaluate_all_methods.py
"""
📊 فصل 4: ارزیابی جامع 5 روش
"""
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.env.environment import UAVMECEnvironment
from models.dag import DAG
from algorithms.baselines.simple_policies import (
    GreedyLocalPolicy,
    AlwaysEdgePolicy,
    AlwaysCloudPolicy
)
from algorithms.baselines.dqn_agent import DQNAgent
from algorithms.baselines.ddpg_agent import DDPGAgent

try:
    from agents.maddpg_agent import MADDPGAgent
    MADDPG_AVAILABLE = True
except ImportError:
    MADDPG_AVAILABLE = False
    print("⚠️  MADDPG not found - will skip")


# ============================================
# 🔧 ENV WRAPPER - تبدیل numpy array به dict
# ============================================
class ActionConverterWrapper:
    """
    Wrapper برای محیط که numpy array رو به dict تبدیل می‌کنه
    """
    def __init__(self, env, num_uavs=3):
        self.env = env
        self.num_uavs = num_uavs
    
    def reset(self, dag=None):
        """Reset environment"""
        if dag is not None:
            return self.env.reset(dag)
        return self.env.reset()
    
    def step(self, action):
        """
        تبدیل numpy array به dict قبل از پاس دادن به محیط
        
        action format (numpy array):
            [dx, dy, dz, offload_decision] * num_uavs
        
        به dict تبدیل میشه:
            {
                'offload': int,
                'cpu': float,
                'bandwidth': [float, float, float],
                'move': [dx, dy]
            }
        """
        if isinstance(action, np.ndarray):
            # تبدیل numpy array به dict
            action_dict = {
                'offload': int(action[3]) if len(action) > 3 else 0,  # offload decision
                'cpu': np.random.rand(),  # placeholder
                'bandwidth': np.random.dirichlet([1, 1, 1]),  # placeholder
                'move': [float(action[0]), float(action[1])]  # dx, dy
            }
            return self.env.step(action_dict)
        else:
            # اگر از قبل dict است
            return self.env.step(action)
    
    def __getattr__(self, name):
        """Forward سایر attributeها به env اصلی"""
        return getattr(self.env, name)


class RandomPolicy:
    """سیاست Random"""
    def __init__(self, num_uavs=3):
        self.num_uavs = num_uavs
    
    def select_action(self, state, evaluation=False):
        """برگرداندن dict"""
        return {
            'offload': np.random.randint(0, 5),
            'cpu': np.random.rand(),
            'bandwidth': np.random.dirichlet([1, 1, 1]),
            'move': np.random.randn(2) * 5
        }


class Chapter4Evaluator:
    """ارزیابی کننده جامع فصل 4"""
    
    def __init__(self, num_episodes=100, max_steps=500, num_tasks=100, num_uavs=3):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.num_tasks = num_tasks
        self.num_uavs = num_uavs
        self.results_dir = Path(project_root) / "results" / "chapter4_evaluation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔧 Initializing environment...")
        base_env = UAVMECEnvironment(device='cpu', max_steps=max_steps)
        
        # ✅ Wrap environment با converter
        self.env = ActionConverterWrapper(base_env, num_uavs=num_uavs)
        
        print(f"🔧 Generating DAG with {num_tasks} tasks...")
        self.dag = DAG.generate_random_dag(
            num_nodes=num_tasks,
            edge_prob=0.3,
            device='cpu'
        )
        print(f"   ✅ DAG created: {self.dag['num_nodes']} nodes, {self.dag['num_edges']} edges")
        
        state = self.env.reset(self.dag)
        if isinstance(state, dict):
            self.state_dim = len(list(state.values())[0])
        else:
            self.state_dim = len(state) if hasattr(state, '__len__') else 537
        
        print(f"✅ State dimension: {self.state_dim}")
        
    def evaluate_method(self, method_name, agent, description):
        """ارزیابی یک روش"""
        print(f"\n{'='*70}")
        print(f"🔍 Evaluating: {method_name}")
        print(f"   Description: {description}")
        print(f"{'='*70}")
        
        episode_rewards = []
        episode_steps = []
        episode_details = []
        
        for episode in tqdm(range(self.num_episodes), desc=f"  {method_name}"):
            state = self.env.reset(self.dag)
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < self.max_steps:
                try:
                    # Get action
                    if hasattr(agent, 'select_action'):
                        action = agent.select_action(state, evaluation=True)
                    else:
                        raise AttributeError(f"Agent has no 'select_action' method")
                    
                    # Step (wrapper خودش تبدیل می‌کنه)
                    result = self.env.step(action)
                    
                    if len(result) == 4:
                        next_state, reward, done, info = result
                    elif len(result) == 5:
                        next_state, reward, terminated, truncated, info = result
                        done = terminated or truncated
                    else:
                        raise ValueError(f"Unexpected step return: {len(result)} values")
                    
                    if isinstance(reward, dict):
                        step_reward = sum(reward.values()) / len(reward)
                    else:
                        step_reward = float(reward)
                    
                    episode_reward += step_reward
                    step_count += 1
                    state = next_state
                    
                except Exception as e:
                    print(f"⚠️  Error at episode {episode}, step {step_count}: {e}")
                    break
            
            episode_rewards.append(episode_reward)
            episode_steps.append(step_count)
            episode_details.append({
                'episode': episode,
                'reward': episode_reward,
                'steps': step_count
            })
        
        results = {
            'method': method_name,
            'description': description,
            'episodes': self.num_episodes,
            'rewards': {
                'mean': float(np.mean(episode_rewards)),
                'std': float(np.std(episode_rewards)),
                'median': float(np.median(episode_rewards)),
                'min': float(np.min(episode_rewards)),
                'max': float(np.max(episode_rewards)),
                'all_values': [float(r) for r in episode_rewards]
            },
            'steps': {
                'mean': float(np.mean(episode_steps)),
                'std': float(np.std(episode_steps)),
                'median': float(np.median(episode_steps))
            },
            'details': episode_details
        }
        
        print(f"   ✅ Reward: {results['rewards']['mean']:.2f} ± {results['rewards']['std']:.2f}")
        print(f"   📊 Steps: {results['steps']['mean']:.1f} ± {results['steps']['std']:.1f}")
        
        return results
    
    def run_complete_evaluation(self):
        """اجرای ارزیابی کامل"""
        
        print("\n" + "="*70)
        print("🎓 CHAPTER 4 - COMPREHENSIVE EVALUATION")
        print("="*70)
        print(f"📊 Environment: UAVMECEnvironment (with ActionConverter)")
        print(f"📏 State dimension: {self.state_dim}")
        print(f"🤖 Number of UAVs: {self.num_uavs}")
        print(f"📈 Episodes per method: {self.num_episodes}")
        print(f"🔄 Max steps per episode: {self.max_steps}")
        print(f"🌐 DAG: {self.dag['num_nodes']} nodes, {self.dag['num_edges']} edges")
        print("="*70)
        
        all_results = {}
        
        # 1️⃣ RANDOM
        print("\n" + "🎲"*35)
        random_agent = RandomPolicy(num_uavs=self.num_uavs)
        all_results['random'] = self.evaluate_method(
            '1-Random', random_agent, 'Random baseline'
        )
        
        # 2️⃣ GREEDY
        print("\n" + "🎯"*35)
        greedy_agent = GreedyLocalPolicy()
        all_results['greedy'] = self.evaluate_method(
            '2-Greedy-Local', greedy_agent, 'Greedy heuristic'
        )
        
        # 3️⃣ ALWAYS CLOUD
        print("\n" + "☁️"*35)
        cloud_agent = AlwaysCloudPolicy()
        all_results['cloud'] = self.evaluate_method(
            '3-Always-Cloud', cloud_agent, 'Always offload to cloud'
        )
        
        # 4️⃣ DQN
        print("\n" + "🤖"*35)
        try:
            dqn_agent = DQNAgent(state_dim=self.state_dim, action_dim=5)
            dqn_path = Path(project_root) / "results" / "baselines" / "dqn" / "dqn_model.pt"
            
            dqn_desc = "DQN with random init"
            if dqn_path.exists():
                try:
                    dqn_agent.load(str(dqn_path))
                    dqn_desc = "DQN with trained weights"
                    print(f"   ✅ Loaded from {dqn_path}")
                except Exception as e:
                    print(f"   ⚠️  Load failed: {e}")
            
            all_results['dqn'] = self.evaluate_method('4-DQN', dqn_agent, dqn_desc)
        except Exception as e:
            print(f"   ⚠️  DQN failed: {e}")
        
        # 5️⃣ DDPG
        print("\n" + "🎯"*35)
        try:
            ddpg_agent = DDPGAgent(state_dim=self.state_dim, action_dim=6)
            ddpg_path = Path(project_root) / "results" / "baselines" / "ddpg" / "ddpg_model.pt"
            
            ddpg_desc = "DDPG with random init"
            if ddpg_path.exists():
                try:
                    ddpg_agent.load(str(ddpg_path))
                    ddpg_desc = "DDPG with trained weights"
                    print(f"   ✅ Loaded from {ddpg_path}")
                except Exception as e:
                    print(f"   ⚠️  Load failed: {e}")
            
            all_results['ddpg'] = self.evaluate_method('5-DDPG', ddpg_agent, ddpg_desc)
        except Exception as e:
            print(f"   ⚠️  DDPG failed: {e}")
        
        # 6️⃣ MADDPG
        if MADDPG_AVAILABLE:
            print("\n" + "⭐"*35)
            checkpoint_paths = [
                Path(project_root) / "output" / "training_runs" / "checkpoints" / "best_ep842.pt",
                Path(project_root) / "results" / "maddpg" / "best_model.pt"
            ]
            
            for ckpt_path in checkpoint_paths:
                if ckpt_path.exists():
                    try:
                        maddpg_agent = MADDPGAgent(
                            num_agents=self.num_uavs,
                            state_dim=self.state_dim,
                            action_dim=6
                        )
                        maddpg_agent.load(str(ckpt_path))
                        all_results['maddpg'] = self.evaluate_method(
                            '6-MADDPG', maddpg_agent, '⭐ MADDPG (Proposed)'
                        )
                        break
                    except Exception as e:
                        print(f"   ⚠️  Load failed: {e}")
        
        self._save_results(all_results)
        self._print_comparison_table(all_results)
        self._generate_plots(all_results)
        
        return all_results
    
    def _save_results(self, results):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.results_dir / f"evaluation_results_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {filepath}")
    
    def _print_comparison_table(self, results):
        print("\n" + "="*90)
        print("📊 FINAL COMPARISON TABLE")
        print("="*90)
        print(f"{'Method':<25} {'Mean Reward':<20} {'Std':<12} {'Steps':<15}")
        print("-"*90)
        
        for key in ['random', 'greedy', 'cloud', 'dqn', 'ddpg', 'maddpg']:
            if key in results:
                data = results[key]
                prefix = "⭐ " if key == 'maddpg' else "   "
                print(f"{prefix}{data['method']:<23} {data['rewards']['mean']:>12.2f}        "
                      f"{data['rewards']['std']:>8.2f}    {data['steps']['mean']:>10.1f}")
        print("="*90)
    
    def _generate_plots(self, results):
        print("\n📈 Generating plots...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Chapter 4: Method Comparison', fontsize=16, fontweight='bold')
        
        methods, means, stds, colors = [], [], [], []
        for key in ['random', 'greedy', 'cloud', 'dqn', 'ddpg', 'maddpg']:
            if key in results:
                methods.append(results[key]['method'])
                means.append(results[key]['rewards']['mean'])
                stds.append(results[key]['rewards']['std'])
                colors.append('gold' if key == 'maddpg' else 'skyblue')
        
        # Plot 1: Bar chart
        axes[0, 0].bar(methods, means, yerr=stds, capsize=5, color=colors, 
                       edgecolor='black', linewidth=1.5)
        axes[0, 0].set_title('Mean Reward ± Std')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].tick_params(axis='x', rotation=15)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Box plot
        reward_data = [results[k]['rewards']['all_values'] 
                      for k in ['random', 'greedy', 'cloud', 'dqn', 'ddpg', 'maddpg'] 
                      if k in results]
        bp = axes[0, 1].boxplot(reward_data, labels=methods, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[0, 1].set_title('Reward Distribution')
        axes[0, 1].tick_params(axis='x', rotation=15)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Learning curves
        color_map = {'random': 'gray', 'greedy': 'blue', 'cloud': 'cyan',
                     'dqn': 'green', 'ddpg': 'orange', 'maddpg': 'red'}
        for key in ['random', 'greedy', 'cloud', 'dqn', 'ddpg', 'maddpg']:
            if key in results:
                axes[1, 0].plot(results[key]['rewards']['all_values'],
                               label=results[key]['method'],
                               color=color_map.get(key, 'black'),
                               linewidth=2 if key=='maddpg' else 1, alpha=0.7)
        axes[1, 0].set_title('Episode Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Summary
        axes[1, 1].axis('off')
        summary = "SUMMARY\n" + "="*40 + "\n\n"
        for key in ['random', 'greedy', 'cloud', 'dqn', 'ddpg', 'maddpg']:
            if key in results:
                data = results[key]
                marker = "⭐ " if key == 'maddpg' else "• "
                summary += f"{marker}{data['method']}:\n"
                summary += f"  {data['rewards']['mean']:.2f} ± {data['rewards']['std']:.2f}\n\n"
        
        axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plot_path = self.results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Plot saved: {plot_path}")
        plt.close()


def main():
    print("\n" + "🎓"*35)
    print("CHAPTER 4: COMPREHENSIVE EVALUATION")
    print("🎓"*35 + "\n")
    
    evaluator = Chapter4Evaluator(
        num_episodes=100,
        max_steps=500,
        num_tasks=100,
        num_uavs=3
    )
    
    results = evaluator.run_complete_evaluation()
    
    print("\n" + "✅"*35)
    print("COMPLETED!")
    print("✅"*35 + "\n")
    
    return results


if __name__ == "__main__":
    main()
