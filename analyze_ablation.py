"""
Ablation Study Visualization - FINAL CORRECTED VERSION
Works with actual JSON structure: episode_rewards, eval_rewards, etc.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'serif'

class AblationAnalyzer:
    def __init__(self, results_dir: str = "results/ablation"):
        self.results_dir = Path(results_dir)
        self.variants = [
            'full_model',
            'no_gat', 
            'no_temporal',
            'decentralized',
            'simpler_arch'
        ]
        self.variant_labels = {
            'full_model': 'Full Model',
            'no_gat': 'No GAT',
            'no_temporal': 'No Temporal',
            'decentralized': 'Decentralized',
            'simpler_arch': 'Simpler Arch'
        }
        self.colors = {
            'full_model': '#2ecc71',
            'no_gat': '#3498db',
            'no_temporal': '#9b59b6',
            'decentralized': '#e74c3c',
            'simpler_arch': '#f39c12'
        }
        self.data = {}
        self.load_data()
        
    def load_data(self):
        """Load all training results"""
        print("📂 Loading data from all variants...")
        for variant in self.variants:
            results_file = self.results_dir / variant / "training_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    self.data[variant] = json.load(f)
                print(f"  ✓ Loaded {variant}")
            else:
                print(f"  ✗ Missing {variant}")
                
    def extract_metrics(self, variant: str) -> Dict:
        """Extract metrics from ACTUAL JSON structure"""
        if variant not in self.data:
            return None
        
        data = self.data[variant]
        
        # Use episode_rewards (NOT training_history)
        if 'episode_rewards' not in data:
            print(f"  ⚠️  No episode_rewards in {variant}")
            return None
            
        rewards = np.array(data['episode_rewards'])
        
        # Calculate statistics
        metrics = {
            'rewards': rewards,
            'best_reward': data.get('best_eval_reward', np.max(rewards)),
            'final_avg': data.get('final_avg_reward', np.mean(rewards[-100:])),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'convergence_episode': self._find_convergence(rewards),
            'stability': np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards),
            'training_time': data.get('training_time_minutes', 0)
        }
        return metrics
        
    def _find_convergence(self, rewards: np.ndarray, window: int = 50) -> int:
        """Find convergence point"""
        if len(rewards) < window * 2:
            return len(rewards)
            
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        moving_std = np.array([np.std(rewards[i:i+window]) 
                               for i in range(len(rewards)-window+1)])
        
        for i in range(window, len(moving_std)):
            if abs(moving_std[i] - moving_std[i-1]) / (moving_std[i-1] + 1e-6) < 0.1:
                return i
        return len(rewards)
    
    def plot_learning_curves(self, save_path: str = "ablation_learning_curves.png"):
        """Plot learning curves"""
        print("\n📈 Creating learning curves plot...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for variant in self.variants:
            metrics = self.extract_metrics(variant)
            if metrics is None:
                continue
                
            rewards = metrics['rewards']
            episodes = np.arange(len(rewards))
            
            # Smoothing
            window = min(20, len(rewards) // 10)
            if window < 2:
                window = 2
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            
            ax.plot(episodes[:len(smoothed)], smoothed, 
                   label=self.variant_labels[variant],
                   color=self.colors[variant],
                   linewidth=2)
            
            # Confidence band
            std = np.array([np.std(rewards[max(0,i-window):i+1]) 
                           for i in range(len(smoothed))])
            ax.fill_between(episodes[:len(smoothed)],
                           smoothed - std,
                           smoothed + std,
                           color=self.colors[variant],
                           alpha=0.15)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, label='Zero Reward')
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Average Reward (smoothed)', fontweight='bold')
        ax.set_title('Learning Curves: Ablation Study Comparison', 
                    fontweight='bold', pad=20)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.results_dir / save_path
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
        
    def plot_performance_comparison(self, save_path: str = "ablation_performance_bar.png"):
        """Bar chart comparison"""
        print("\n📊 Creating performance comparison bar chart...")
        
        metrics = {var: self.extract_metrics(var) for var in self.variants 
                  if var in self.data and self.extract_metrics(var) is not None}
        
        if not metrics:
            print("  ⚠️  No valid metrics to plot!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Best Reward
        variants_sorted = sorted(metrics.keys(), 
                               key=lambda x: metrics[x]['best_reward'], 
                               reverse=True)
        best_rewards = [metrics[v]['best_reward'] for v in variants_sorted]
        colors_sorted = [self.colors[v] for v in variants_sorted]
        labels_sorted = [self.variant_labels[v] for v in variants_sorted]
        
        bars1 = ax1.bar(range(len(variants_sorted)), best_rewards, 
                       color=colors_sorted, edgecolor='black', linewidth=1.5)
        ax1.set_xticks(range(len(variants_sorted)))
        ax1.set_xticklabels(labels_sorted, rotation=45, ha='right')
        ax1.set_ylabel('Best Reward', fontweight='bold')
        ax1.set_title('Best Reward Achieved', fontweight='bold', pad=15)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, best_rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom' if val > 0 else 'top',
                    fontweight='bold', fontsize=9)
        
        # Final Average
        final_avgs = [metrics[v]['final_avg'] for v in variants_sorted]
        bars2 = ax2.bar(range(len(variants_sorted)), final_avgs,
                       color=colors_sorted, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(variants_sorted)))
        ax2.set_xticklabels(labels_sorted, rotation=45, ha='right')
        ax2.set_ylabel('Final Avg Reward (Last 100)', fontweight='bold')
        ax2.set_title('Final Performance', fontweight='bold', pad=15)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, final_avgs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom' if val > 0 else 'top',
                    fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        output_path = self.results_dir / save_path
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
        
    def plot_distribution_boxplot(self, save_path: str = "ablation_distribution.png"):
        """Distribution box plot"""
        print("\n📦 Creating distribution box plot...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data_for_plot = []
        labels = []
        colors_list = []
        
        for variant in self.variants:
            metrics = self.extract_metrics(variant)
            if metrics is None:
                continue
            n_episodes = min(200, len(metrics['rewards']))
            data_for_plot.append(metrics['rewards'][-n_episodes:])
            labels.append(self.variant_labels[variant])
            colors_list.append(self.colors[variant])
        
        if not data_for_plot:
            print("  ⚠️  No data to plot!")
            return
        
        bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(color='red', linewidth=2),
                       meanprops=dict(color='blue', linewidth=2, linestyle='--'))
        
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_ylabel('Reward Distribution (Last 200 eps)', fontweight='bold')
        ax.set_title('Reward Distribution Comparison', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Median'),
            Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        output_path = self.results_dir / save_path
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
        
    def plot_component_importance(self, save_path: str = "ablation_importance.png"):
        """Component importance heatmap"""
        print("\n🔥 Creating component importance heatmap...")
        
        full_metrics = self.extract_metrics('full_model')
        if full_metrics is None:
            print("  ⚠️  Cannot load full_model!")
            return
            
        full_best = full_metrics['best_reward']
        full_final = full_metrics['final_avg']
        
        impacts = {}
        for variant in self.variants:
            if variant == 'full_model':
                continue
            metrics = self.extract_metrics(variant)
            if metrics is None:
                continue
            impacts[variant] = {
                'Best Reward Impact': full_best - metrics['best_reward'],
                'Final Avg Impact': full_final - metrics['final_avg'],
                'Stability Impact': metrics['stability'] - full_metrics['stability']
            }
        
        if not impacts:
            print("  ⚠️  No impacts to plot!")
            return
        
        df = pd.DataFrame(impacts).T
        df.index = [self.variant_labels[v] for v in df.index]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                   center=0, cbar_kws={'label': 'Performance Degradation'},
                   linewidths=1, linecolor='white', ax=ax)
        ax.set_title('Component Importance\n(Higher = More Important)', 
                    fontweight='bold', pad=20)
        ax.set_ylabel('')
        
        plt.tight_layout()
        output_path = self.results_dir / save_path
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_path}")
        plt.close()
        
    def statistical_analysis(self, save_path: str = "statistical_analysis.txt"):
        """Statistical tests"""
        print("\n📊 Performing statistical analysis...")
        
        metrics = {var: self.extract_metrics(var) for var in self.variants 
                  if var in self.data and self.extract_metrics(var) is not None}
        
        if not metrics:
            print("  ⚠️  No valid metrics!")
            return
        
        output = []
        output.append("=" * 70)
        output.append("STATISTICAL ANALYSIS OF ABLATION STUDY")
        output.append("=" * 70)
        output.append("")
        
        output.append("## SUMMARY STATISTICS")
        output.append("-" * 70)
        for variant in self.variants:
            if variant not in metrics:
                continue
            m = metrics[variant]
            output.append(f"\n{self.variant_labels[variant]}:")
            output.append(f"  Best Reward:      {m['best_reward']:>10.2f}")
            output.append(f"  Final Avg:        {m['final_avg']:>10.2f}")
            output.append(f"  Mean Reward:      {m['mean_reward']:>10.2f}")
            output.append(f"  Std Dev:          {m['std_reward']:>10.2f}")
            output.append(f"  Stability:        {m['stability']:>10.2f}")
            output.append(f"  Convergence Ep:   {m['convergence_episode']:>10d}")
            output.append(f"  Training Time:    {m['training_time']:>10.1f} min")
        
        output.append("\n" + "=" * 70)
        output.append("## PAIRWISE T-TESTS (vs Full Model)")
        output.append("-" * 70)
        
        if 'full_model' not in metrics:
            output.append("\n⚠️  No full_model for comparison")
        else:
            full_rewards = metrics['full_model']['rewards'][-100:]
            
            for variant in self.variants:
                if variant == 'full_model' or variant not in metrics:
                    continue
                    
                var_rewards = metrics[variant]['rewards'][-100:]
                t_stat, p_value = stats.ttest_ind(full_rewards, var_rewards)
                
                pooled_std = np.sqrt((np.std(full_rewards)**2 + np.std(var_rewards)**2) / 2)
                cohen_d = (np.mean(full_rewards) - np.mean(var_rewards)) / (pooled_std + 1e-10)
                
                output.append(f"\n{self.variant_labels[variant]}:")
                output.append(f"  t-statistic:      {t_stat:>10.4f}")
                output.append(f"  p-value:          {p_value:>10.4e}")
                output.append(f"  Significant:      {'Yes' if p_value < 0.05 else 'No':>10s}")
                output.append(f"  Cohen's d:        {cohen_d:>10.4f}")
                
                if abs(cohen_d) < 0.2:
                    effect = "negligible"
                elif abs(cohen_d) < 0.5:
                    effect = "small"
                elif abs(cohen_d) < 0.8:
                    effect = "medium"
                else:
                    effect = "large"
                output.append(f"  Effect size:      {effect:>10s}")
        
        output_text = "\n".join(output)
        output_file = self.results_dir / save_path
        with open(output_file, 'w') as f:
            f.write(output_text)
        
        print(f"  ✓ Saved to {output_file}")
        print("\n" + output_text)
        
    def generate_latex_table(self, save_path: str = "ablation_table.tex"):
        """LaTeX table"""
        print("\n📄 Generating LaTeX table...")
        
        metrics = {var: self.extract_metrics(var) for var in self.variants 
                  if var in self.data and self.extract_metrics(var) is not None}
        
        if not metrics:
            print("  ⚠️  No metrics for LaTeX!")
            return
        
        variants_sorted = sorted(metrics.keys(), 
                               key=lambda x: metrics[x]['best_reward'], 
                               reverse=True)
        
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("\\centering")
        latex.append("\\caption{Ablation Study Results}")
        latex.append("\\label{tab:ablation}")
        latex.append("\\begin{tabular}{lccccc}")
        latex.append("\\toprule")
        latex.append("Variant & Best & Final Avg & Stability & Conv. Ep & Time (min) \\\\")
        latex.append("\\midrule")
        
        for variant in variants_sorted:
            m = metrics[variant]
            label = self.variant_labels[variant]
            
            if variant == variants_sorted[0]:
                latex.append(f"\\textbf{{{label}}} & \\textbf{{{m['best_reward']:.2f}}} & "
                           f"\\textbf{{{m['final_avg']:.2f}}} & {m['stability']:.2f} & "
                           f"{m['convergence_episode']} & {m['training_time']:.1f} \\\\")
            else:
                latex.append(f"{label} & {m['best_reward']:.2f} & {m['final_avg']:.2f} & "
                           f"{m['stability']:.2f} & {m['convergence_episode']} & "
                           f"{m['training_time']:.1f} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        latex_text = "\n".join(latex)
        output_file = self.results_dir / save_path
        with open(output_file, 'w') as f:
            f.write(latex_text)
        
        print(f"  ✓ Saved to {output_file}")
        print("\n" + latex_text)
        
    def run_full_analysis(self):
        """Run complete analysis"""
        print("\n" + "="*70)
        print("🚀 RUNNING COMPLETE ABLATION STUDY ANALYSIS")
        print("="*70)
        
        self.plot_learning_curves()
        self.plot_performance_comparison()
        self.plot_distribution_boxplot()
        self.plot_component_importance()
        self.statistical_analysis()
        self.generate_latex_table()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\n📁 All outputs: {self.results_dir.absolute()}")


if __name__ == "__main__":
    analyzer = AblationAnalyzer("results/ablation")
    analyzer.run_full_analysis()
