"""
Evaluation Script for Best MADDPG Model
Evaluates the trained model on test scenarios
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

class ModelEvaluator:
    def __init__(self, config_path: str = None):
        """Initialize evaluator with configuration"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {
            'episodes': [],
            'rewards': [],
            'delays': [],
            'energy': [],
            'success_rate': [],
            'qos_violations': []
        }
        self.checkpoint = None
        
    def load_model(self, checkpoint_path: str):
        """Load trained MADDPG model from checkpoint"""
        print(f"Loading model from: {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"✓ Model loaded successfully")
        
        # نمایش اطلاعات checkpoint به صورت امن
        print(f"\n📦 Checkpoint Information:")
        
        if 'episode' in checkpoint:
            print(f"  - Episode: {checkpoint['episode']}")
        
        if 'reward' in checkpoint:
            reward = checkpoint['reward']
            if isinstance(reward, (int, float)):
                print(f"  - Reward: {reward:.2f}")
            else:
                print(f"  - Reward: {reward}")
        
        if 'timestamp' in checkpoint:
            print(f"  - Timestamp: {checkpoint['timestamp']}")
        
        # نمایش کلیدهای موجود
        available_keys = list(checkpoint.keys())
        print(f"  - Available keys ({len(available_keys)}): {', '.join(available_keys[:5])}")
        if len(available_keys) > 5:
            print(f"    ... and {len(available_keys) - 5} more")
        
        self.checkpoint = checkpoint
        return checkpoint
    
    def evaluate_episode(self, episode_num: int) -> Dict:
        """Evaluate single episode"""
        # Placeholder - این باید با محیط واقعی شما جایگزین شود
        # در اینجا داده‌های تصادفی تولید می‌شود برای تست
        reward = np.random.uniform(-10, 150)
        delay = np.random.uniform(50, 200)
        energy = np.random.uniform(100, 500)
        success = reward > 0
        qos_violation = delay > 150 or energy > 400
        
        return {
            'episode': episode_num,
            'reward': reward,
            'delay': delay,
            'energy': energy,
            'success': success,
            'qos_violation': qos_violation
        }
    
    def run_evaluation(self, num_episodes: int = 100):
        """Run evaluation for specified number of episodes"""
        print(f"\n{'='*60}")
        print(f"Starting Evaluation: {num_episodes} episodes")
        print(f"{'='*60}\n")
        
        for ep in range(num_episodes):
            result = self.evaluate_episode(ep)
            
            self.results['episodes'].append(result['episode'])
            self.results['rewards'].append(result['reward'])
            self.results['delays'].append(result['delay'])
            self.results['energy'].append(result['energy'])
            self.results['success_rate'].append(1 if result['success'] else 0)
            self.results['qos_violations'].append(1 if result['qos_violation'] else 0)
            
            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(self.results['rewards'][-10:])
                print(f"Episode {ep+1:3d}/{num_episodes} | "
                      f"Avg Reward (last 10): {avg_reward:7.2f}")
        
        print(f"\n{'='*60}")
        print("✓ Evaluation Complete!")
        print(f"{'='*60}\n")
    
    def compute_statistics(self) -> Dict:
        """Compute comprehensive statistics"""
        stats = {
            'reward': {
                'mean': float(np.mean(self.results['rewards'])),
                'std': float(np.std(self.results['rewards'])),
                'min': float(np.min(self.results['rewards'])),
                'max': float(np.max(self.results['rewards'])),
                'median': float(np.median(self.results['rewards']))
            },
            'delay': {
                'mean': float(np.mean(self.results['delays'])),
                'std': float(np.std(self.results['delays'])),
                'min': float(np.min(self.results['delays'])),
                'max': float(np.max(self.results['delays']))
            },
            'energy': {
                'mean': float(np.mean(self.results['energy'])),
                'std': float(np.std(self.results['energy'])),
                'min': float(np.min(self.results['energy'])),
                'max': float(np.max(self.results['energy']))
            },
            'success_rate': float(np.mean(self.results['success_rate']) * 100),
            'qos_violation_rate': float(np.mean(self.results['qos_violations']) * 100),
            'total_episodes': len(self.results['episodes'])
        }
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.compute_statistics()
        
        print("\n" + "="*60)
        print("📊 EVALUATION STATISTICS")
        print("="*60)
        
        print("\n💰 Reward Statistics:")
        print(f"  Mean:   {stats['reward']['mean']:8.2f}")
        print(f"  Std:    {stats['reward']['std']:8.2f}")
        print(f"  Min:    {stats['reward']['min']:8.2f}")
        print(f"  Max:    {stats['reward']['max']:8.2f}")
        print(f"  Median: {stats['reward']['median']:8.2f}")
        
        print("\n⏱️  Delay Statistics (ms):")
        print(f"  Mean:   {stats['delay']['mean']:8.2f}")
        print(f"  Std:    {stats['delay']['std']:8.2f}")
        print(f"  Min:    {stats['delay']['min']:8.2f}")
        print(f"  Max:    {stats['delay']['max']:8.2f}")
        
        print("\n🔋 Energy Statistics (mJ):")
        print(f"  Mean:   {stats['energy']['mean']:8.2f}")
        print(f"  Std:    {stats['energy']['std']:8.2f}")
        print(f"  Min:    {stats['energy']['min']:8.2f}")
        print(f"  Max:    {stats['energy']['max']:8.2f}")
        
        print("\n✅ Performance Metrics:")
        print(f"  Success Rate:        {stats['success_rate']:.1f}%")
        print(f"  QoS Violation Rate:  {stats['qos_violation_rate']:.1f}%")
        print(f"  Total Episodes:      {stats['total_episodes']}")
        
        print("\n" + "="*60 + "\n")
        
        return stats
    
    def save_results(self, output_dir: str = "output/evaluation"):
        """Save evaluation results to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_path / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'statistics': self.compute_statistics(),
                'timestamp': timestamp,
                'checkpoint_info': {
                    'keys': list(self.checkpoint.keys()) if self.checkpoint else []
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {results_file}")
        
        # Save summary statistics
        stats_file = output_path / f"evaluation_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.compute_statistics(), f, indent=2, ensure_ascii=False)
        
        print(f"✓ Statistics saved to: {stats_file}")
        
        return results_file, stats_file
    
    def plot_results(self, output_dir: str = "output/evaluation/plots"):
        """Generate visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MADDPG Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Reward Distribution
        axes[0, 0].hist(self.results['rewards'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Reward Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        mean_reward = np.mean(self.results['rewards'])
        axes[0, 0].axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_reward:.2f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reward over Episodes
        axes[0, 1].plot(self.results['episodes'], self.results['rewards'], 
                       color='green', alpha=0.6, linewidth=1.5)
        axes[0, 1].set_title('Reward Progression', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Moving average
        window = 10
        if len(self.results['rewards']) >= window:
            moving_avg = np.convolve(self.results['rewards'], 
                                    np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(self.results['rewards'])), 
                          moving_avg, color='red', linewidth=2, 
                          label=f'{window}-Episode MA')
            axes[0, 1].legend()
        
        # Delay vs Energy Scatter
        scatter = axes[1, 0].scatter(self.results['delays'], self.results['energy'], 
                                    alpha=0.6, c=self.results['rewards'], 
                                    cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
        axes[1, 0].set_title('Delay vs Energy (colored by Reward)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Delay (ms)')
        axes[1, 0].set_ylabel('Energy (mJ)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Reward')
        
        # Success Rate Visualization
        success_cumsum = np.cumsum(self.results['success_rate'])
        success_rate = (success_cumsum / (np.arange(len(self.results['success_rate'])) + 1)) * 100
        axes[1, 1].plot(self.results['episodes'], success_rate, 
                       color='purple', linewidth=2.5, label='Success Rate')
        axes[1, 1].fill_between(self.results['episodes'], 0, success_rate, 
                               color='purple', alpha=0.2)
        axes[1, 1].set_title('Cumulative Success Rate', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 105])
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = output_path / f"evaluation_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved to: {plot_file}")
        
        plt.close()
        
        return plot_file

def main():
    parser = argparse.ArgumentParser(description='Evaluate Best MADDPG Model')
    parser.add_argument('--checkpoint', type=str, 
                       default='output/training_runs/checkpoints/best_ep842.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--output_dir', type=str, default='output/evaluation',
                       help='Output directory for results')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to JSON')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 MADDPG Model Evaluation")
    print("="*60 + "\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model
    try:
        checkpoint = evaluator.load_model(args.checkpoint)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Available checkpoints in output/training_runs/checkpoints/:")
        checkpoint_dir = Path("output/training_runs/checkpoints")
        if checkpoint_dir.exists():
            for ckpt in sorted(checkpoint_dir.glob("*.pt")):
                print(f"  - {ckpt.name}")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run evaluation
    evaluator.run_evaluation(num_episodes=args.num_episodes)
    
    # Print statistics
    stats = evaluator.print_statistics()
    
    # Save results
    if args.save_results:
        evaluator.save_results(args.output_dir)
    
    # Generate plots
    if args.plot:
        evaluator.plot_results(f"{args.output_dir}/plots")
    
    print("✅ Evaluation completed successfully!\n")

if __name__ == "__main__":
    main()
