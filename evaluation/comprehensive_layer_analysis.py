"""
comprehensive_layer_analysis.py
تحلیل جامع عملکرد MADDPG در لایه‌های مختلف محاسباتی
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrainedActorNetwork(nn.Module):
    """
    Actor Network با معماری دقیق checkpoint:
    - 3 لایه FC با LayerNorm
    - Hidden: 512 → 512 → 256
    - فقط offload_head آموزش دیده
    """
    
    def __init__(self, state_dim=537, offload_dim=5, hidden=512):
        super().__init__()
        
        # Layer 1: 537 → 512
        self.fc1 = nn.Linear(state_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        
        # Layer 2: 512 → 512
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        
        # Layer 3: 512 → 256
        self.fc3 = nn.Linear(hidden, 256)
        self.ln3 = nn.LayerNorm(256)
        
        # Output heads
        self.offload_head = nn.Linear(256, offload_dim)
        
        # NOTE: continuous_head در checkpoint هست اما با dimension=0
        # ما آن را load نمی‌کنیم
        
        self.activation = nn.ELU()
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: (B, 537)
        
        Returns:
            offload_logits: (B, 5)
        """
        x = self.activation(self.ln1(self.fc1(state)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.activation(self.ln3(self.fc3(x)))
        
        offload_logits = self.offload_head(x)
        
        return offload_logits


class HeuristicContinuousActions:
    """
    استراتژی‌های Heuristic برای continuous actions
    (چون مدل این بخش را یاد نگرفته)
    """
    
    def __init__(self, strategy='balanced'):
        """
        Strategies:
        - balanced: توزیع متعادل منابع
        - greedy: حداکثر استفاده از منابع
        - conservative: حداقل استفاده
        - adaptive: بر اساس وضعیت سیستم
        """
        self.strategy = strategy
    
    def generate(self, offload_choice: int, state: np.ndarray) -> Dict:
        """
        تولید continuous actions بر اساس استراتژی
        
        Args:
            offload_choice: 0=Local, 1=Edge, 2=Fog, 3=Cloud, 4=Reject
            state: vector وضعیت (537,)
        
        Returns:
            dict with 'cpu', 'bandwidth', 'move'
        """
        
        if self.strategy == 'balanced':
            cpu = 0.5
            bandwidth = np.array([0.33, 0.33, 0.34])
            move = np.array([0.0, 0.0])
            
        elif self.strategy == 'greedy':
            cpu = 0.9 if offload_choice != 4 else 0.1
            bandwidth = np.array([0.7, 0.2, 0.1]) if offload_choice in [1,2] else np.array([0.33, 0.33, 0.34])
            move = np.array([0.0, 0.0])
            
        elif self.strategy == 'conservative':
            cpu = 0.3
            bandwidth = np.array([0.5, 0.3, 0.2])
            move = np.array([0.0, 0.0])
            
        elif self.strategy == 'adaptive':
            # استفاده از اطلاعات state
            # فرض: state[0:5] → battery levels
            battery_avg = state[0:5].mean() if len(state) >= 5 else 0.5
            
            cpu = 0.3 if battery_avg < 0.3 else 0.7
            bandwidth = np.array([0.4, 0.4, 0.2]) if battery_avg > 0.5 else np.array([0.6, 0.2, 0.2])
            move = np.array([0.0, 0.0])
        
        else:
            cpu = 0.5
            bandwidth = np.array([0.33, 0.33, 0.34])
            move = np.array([0.0, 0.0])
        
        return {
            'cpu': float(cpu),
            'bandwidth': bandwidth,
            'move': move
        }


class LayerMetricsCalculator:
    """
    محاسبه معیارهای عملکرد برای هر لایه محاسباتی
    """
    
    def __init__(self):
        self.layer_names = {
            0: 'Local',
            1: 'Edge', 
            2: 'Fog',
            3: 'Cloud',
            4: 'Reject'
        }
        
        # پارامترهای شبیه‌سازی
        self.battery_consumption = {
            0: 0.05,  # Local: متوسط
            1: 0.03,  # Edge: کم
            2: 0.04,  # Fog: کم-متوسط
            3: 0.08,  # Cloud: زیاد
            4: 0.01   # Reject: خیلی کم
        }
        
        self.latency_base = {
            0: 10,    # Local: 10ms
            1: 30,    # Edge: 30ms
            2: 50,    # Fog: 50ms
            3: 100,   # Cloud: 100ms
            4: 0      # Reject: 0
        }
    
    def calculate_metrics(self, offload_choice: int, cpu: float, 
                         bandwidth: np.ndarray, complexity: str) -> Dict:
        """
        محاسبه معیارها برای یک تصمیم offloading
        
        Args:
            offload_choice: 0-4
            cpu: [0,1]
            bandwidth: (3,) sum=1
            complexity: 'easy', 'medium', 'hard'
        
        Returns:
            dict with metrics
        """
        
        layer = self.layer_names[offload_choice]
        
        # Battery consumption
        battery = self.battery_consumption[offload_choice]
        battery *= (1.0 + cpu * 0.5)  # CPU usage تاثیر
        
        # Latency
        latency = self.latency_base[offload_choice]
        
        # تاثیر پیچیدگی
        complexity_factor = {'easy': 0.7, 'medium': 1.0, 'hard': 1.5}[complexity]
        latency *= complexity_factor
        
        # تاثیر bandwidth (فقط برای remote layers)
        if offload_choice in [1, 2, 3]:
            bw_efficiency = bandwidth[0]  # اولین کانال
            latency *= (2.0 - bw_efficiency)  # بیشتر BW → کمتر latency
        
        # Overload probability
        overload_base = {'easy': 0.05, 'medium': 0.15, 'hard': 0.30}[complexity]
        
        if offload_choice == 0:  # Local
            overload = overload_base * 1.5  # Local پرمخاطره‌تر
        elif offload_choice == 3:  # Cloud
            overload = overload_base * 0.7  # Cloud پایدارتر
        elif offload_choice == 4:  # Reject
            overload = 0.0
        else:
            overload = overload_base
        
        # Success rate
        success = 1.0 if offload_choice != 4 else 0.0
        
        return {
            'layer': layer,
            'battery_consumption': battery,
            'latency_ms': latency,
            'overload_prob': overload,
            'success': success,
            'cpu_usage': cpu
        }


class LayerAnalysisEvaluator:
    """
    ارزیابی جامع در لایه‌های مختلف
    """
    
    def __init__(self, checkpoint_path: str, strategy: str = 'balanced'):
        self.checkpoint_path = Path(checkpoint_path)
        self.strategy = strategy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.config = self.checkpoint['config']
        
        # Initialize components
        self.actors = self._load_actors()
        self.heuristic = HeuristicContinuousActions(strategy=strategy)
        self.metrics_calc = LayerMetricsCalculator()
        
        print(f"✅ Loaded {len(self.actors)} actors from checkpoint")
        print(f"📊 Config: {self.config['n_agents']} agents, State Dim: {self.config['state_dim']}")
        print(f"🎯 Using '{strategy}' strategy for continuous actions\n")
    
    def _load_actors(self) -> List[nn.Module]:
        """Load trained actor networks"""
        actors = []
        for i, actor_state in enumerate(self.checkpoint['actors']):
            actor = TrainedActorNetwork(
                state_dim=self.config['state_dim'],
                offload_dim=5,
                hidden=self.config['hidden_dim']
            )
            
            # Load با strict=False برای نادیده گرفتن continuous_head
            missing_keys, unexpected_keys = actor.load_state_dict(actor_state, strict=False)
            
            if i == 0 and unexpected_keys:
                print(f"⚠️  Ignoring keys in checkpoint: {unexpected_keys}")
            
            actor.to(self.device)
            actor.eval()
            actors.append(actor)
        
        return actors
    
    def generate_test_scenarios(self, n_scenarios: int = 100) -> List[Dict]:
        """
        تولید سناریوهای تست برای لایه‌ها و پیچیدگی‌های مختلف
        """
        scenarios = []
        complexities = ['easy', 'medium', 'hard']
        
        for _ in range(n_scenarios):
            # Generate random state
            state = np.random.randn(self.config['state_dim']).astype(np.float32)
            
            # Random complexity
            complexity = np.random.choice(complexities)
            
            scenarios.append({
                'state': state,
                'complexity': complexity
            })
        
        return scenarios
    
    def evaluate_scenarios(self, scenarios: List[Dict]) -> pd.DataFrame:
        """
        ارزیابی سناریوها و جمع‌آوری metrics
        """
        results = []
        
        print(f"🔄 Evaluating {len(scenarios)} scenarios...")
        
        for idx, scenario in enumerate(scenarios):
            state = torch.FloatTensor(scenario['state']).unsqueeze(0).to(self.device)
            complexity = scenario['complexity']
            
            # Get decisions from all agents
            with torch.no_grad():
                for agent_id, actor in enumerate(self.actors):
                    # Get offload decision
                    offload_logits = actor(state)
                    offload_choice = torch.argmax(offload_logits, dim=-1).item()
                    
                    # Get heuristic continuous actions
                    cont_actions = self.heuristic.generate(
                        offload_choice, 
                        scenario['state']
                    )
                    
                    # Calculate metrics
                    metrics = self.metrics_calc.calculate_metrics(
                        offload_choice,
                        cont_actions['cpu'],
                        cont_actions['bandwidth'],
                        complexity
                    )
                    
                    # Store result
                    results.append({
                        'scenario_id': idx,
                        'agent_id': agent_id,
                        'complexity': complexity,
                        'offload_choice': offload_choice,
                        **metrics,
                        **{f'bw_{i}': cont_actions['bandwidth'][i] for i in range(3)}
                    })
            
            if (idx + 1) % 20 == 0:
                print(f"   Processed {idx+1}/{len(scenarios)} scenarios")
        
        df = pd.DataFrame(results)
        print(f"✅ Evaluation complete! Generated {len(df)} data points\n")
        
        return df
    
    def analyze_by_layer_and_complexity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تحلیل آماری بر اساس لایه و پیچیدگی
        """
        summary = df.groupby(['layer', 'complexity']).agg({
            'battery_consumption': ['mean', 'std'],
            'latency_ms': ['mean', 'std'],
            'overload_prob': ['mean', 'std'],
            'success': ['mean', 'sum'],  # mean=rate, sum=count
            'cpu_usage': ['mean', 'std']
        }).round(4)
        
        # Flatten multi-index columns
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        return summary
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str = 'evaluation_results'):
        """
        ایجاد نمودارهای تحلیلی
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"📊 Creating visualizations in '{output_dir}/'...")
        
        # 1. Distribution of Offload Choices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, complexity in enumerate(['easy', 'medium', 'hard']):
            subset = df[df['complexity'] == complexity]
            counts = subset['layer'].value_counts()
            
            axes[idx].bar(counts.index, counts.values, color=sns.color_palette("husl", len(counts)))
            axes[idx].set_title(f'Offload Distribution - {complexity.upper()}', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Layer')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'offload_distribution.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: offload_distribution.png")
        plt.close()
        
        # 2. Metrics Heatmap
        summary = self.analyze_by_layer_and_complexity(df)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics_to_plot = [
            ('battery_consumption_mean', 'Battery Consumption'),
            ('latency_ms_mean', 'Latency (ms)'),
            ('overload_prob_mean', 'Overload Probability'),
            ('success_mean', 'Success Rate')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            pivot = summary.pivot(index='layer', columns='complexity', values=metric)
            
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, 
                       cbar_kws={'label': title})
            ax.set_title(f'{title} by Layer & Complexity', fontsize=12, fontweight='bold')
            ax.set_xlabel('Complexity')
            ax.set_ylabel('Layer')
        
        plt.tight_layout()
        plt.savefig(output_path / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: metrics_heatmap.png")
        plt.close()
        
        # 3. Box plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        for idx, (metric, title) in enumerate([
            ('battery_consumption', 'Battery Consumption'),
            ('latency_ms', 'Latency (ms)'),
            ('overload_prob', 'Overload Probability'),
            ('cpu_usage', 'CPU Usage')
        ]):
            ax = axes[idx // 2, idx % 2]
            
            df.boxplot(column=metric, by='layer', ax=ax)
            ax.set_title(f'{title} Distribution by Layer', fontsize=12, fontweight='bold')
            ax.set_xlabel('Layer')
            ax.set_ylabel(title)
            plt.sca(ax)
            plt.xticks(rotation=0)
        
        plt.suptitle('')  # Remove default title
        plt.tight_layout()
        plt.savefig(output_path / 'metrics_boxplot.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: metrics_boxplot.png")
        plt.close()
        
        print("✅ All visualizations created!\n")
    
    def generate_report(self, df: pd.DataFrame, output_dir: str = 'evaluation_results'):
        """
        تولید گزارش نهایی
        """
        output_path = Path(output_dir)
        
        summary = self.analyze_by_layer_and_complexity(df)
        
        report = {
            'checkpoint': str(self.checkpoint_path),
            'strategy': self.strategy,
            'n_agents': self.config['n_agents'],
            'total_scenarios': len(df) // self.config['n_agents'],
            'total_decisions': len(df),
            
            'overall_metrics': {
                'avg_battery': float(df['battery_consumption'].mean()),
                'avg_latency': float(df['latency_ms'].mean()),
                'avg_overload_prob': float(df['overload_prob'].mean()),
                'success_rate': float(df['success'].mean()),
                'completion_rate': float((df['success'] == 1).sum() / len(df))
            },
            
            'layer_distribution': df['layer'].value_counts().to_dict(),
            'complexity_distribution': df['complexity'].value_counts().to_dict(),
            
            'summary_by_layer_complexity': summary.to_dict(orient='records')
        }
        
        # Save JSON
        with open(output_path / 'layer_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save CSV
        df.to_csv(output_path / 'detailed_results.csv', index=False)
        summary.to_csv(output_path / 'summary_statistics.csv', index=False)
        
        print(f"📄 Report saved to '{output_dir}/':")
        print(f"   • layer_analysis_report.json")
        print(f"   • detailed_results.csv")
        print(f"   • summary_statistics.csv\n")
        
        # Print summary
        print("="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70)
        print(f"Total Scenarios:     {report['total_scenarios']}")
        print(f"Total Decisions:     {report['total_decisions']}")
        print(f"Strategy:            {self.strategy}")
        print(f"\n{'Metric':<25} {'Value':<15}")
        print("-"*40)
        print(f"{'Avg Battery Consumption':<25} {report['overall_metrics']['avg_battery']:.4f}")
        print(f"{'Avg Latency (ms)':<25} {report['overall_metrics']['avg_latency']:.2f}")
        print(f"{'Avg Overload Prob':<25} {report['overall_metrics']['avg_overload_prob']:.4f}")
        print(f"{'Success Rate':<25} {report['overall_metrics']['success_rate']:.2%}")
        print(f"{'Completion Rate':<25} {report['overall_metrics']['completion_rate']:.2%}")
        print("="*70)
        
        return report


def main():
    """
    Main execution
    """
    print("\n" + "="*70)
    print("🚀 COMPREHENSIVE LAYER ANALYSIS FOR MADDPG UAV OFFLOADING")
    print("="*70 + "\n")
    
    checkpoint_path = "checkpoints/maddpg/best_model.pt"
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Test different strategies
    strategies = ['balanced', 'greedy', 'conservative', 'adaptive']
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"🧪 Testing Strategy: {strategy.upper()}")
        print(f"{'='*70}\n")
        
        output_dir = f"evaluation_results/layer_analysis_{strategy}"
        
        evaluator = LayerAnalysisEvaluator(checkpoint_path, strategy=strategy)
        
        scenarios = evaluator.generate_test_scenarios(n_scenarios=100)
        
        df = evaluator.evaluate_scenarios(scenarios)
        
        evaluator.create_visualizations(df, output_dir=output_dir)
        
        report = evaluator.generate_report(df, output_dir=output_dir)
        
        print(f"\n✅ Strategy '{strategy}' complete!\n")
    
    print("\n" + "="*70)
    print("🎉 ALL ANALYSES COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
