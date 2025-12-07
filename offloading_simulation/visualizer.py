"""
تولید نمودارهای حرفه‌ای برای نتایج Offloading
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict
import json

# تنظیمات استایل
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class OffloadingVisualizer:
    """کلاس تولید نمودار"""
    
    def __init__(self, output_dir: str = "results/offloading_results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # پالت رنگی
        self.colors = {
            "ground": "#e74c3c",
            "edge": "#3498db",
            "fog": "#2ecc71",
            "cloud": "#9b59b6"
        }
    
    def plot_radar_chart(self, metrics: Dict, complexity: str, save_path: str = None):
        """
        نمودار راداری برای مقایسه لایه‌ها
        """
        layers = ["ground", "edge", "fog", "cloud"]
        
        # استخراج داده‌ها
        categories = ['Scalability', 'Energy Eff.', 'Throughput', 'Latency Red.']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for layer in layers:
            if layer in metrics:
                layer_metrics = metrics[layer]
                
                # نرمال‌سازی مقادیر به [0, 100]
                values = [
                    layer_metrics.get('scalability', 0),
                    100 - layer_metrics.get('energy_mean', 100),  # معکوس برای Energy
                    layer_metrics.get('throughput', 0) * 10,  # Scale
                    layer_metrics.get('latency_reduction', 0)
                ]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=layer.capitalize(), color=self.colors[layer])
                ax.fill(angles, values, alpha=0.15, color=self.colors[layer])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title(f'Performance Comparison - {complexity.capitalize()} Tasks',
                    size=16, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Radar chart saved: {save_path}")
        
        plt.close()
    
    def plot_heatmap(self, all_metrics: Dict, save_path: str = None):
        """
        Heatmap مقایسه‌ای بین لایه‌ها و سطوح پیچیدگی
        """
        complexities = ['simple', 'medium', 'complex']
        layers = ['ground', 'edge', 'fog', 'cloud']
        
        # ماتریس Scalability
        scalability_matrix = np.zeros((len(complexities), len(layers)))
        
        for i, comp in enumerate(complexities):
            if comp in all_metrics:
                for j, layer in enumerate(layers):
                    if layer in all_metrics[comp]:
                        scalability_matrix[i, j] = all_metrics[comp][layer].get('scalability', 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(scalability_matrix, annot=True, fmt='.1f', 
                   xticklabels=[l.capitalize() for l in layers],
                   yticklabels=[c.capitalize() for c in complexities],
                   cmap='RdYlGn', vmin=0, vmax=100, ax=ax,
                   cbar_kws={'label': 'Success Rate (%)'})
        
        ax.set_title('Scalability Heatmap: Success Rate by Layer & Complexity',
                    size=14, weight='bold')
        ax.set_xlabel('Computational Layer', size=12)
        ax.set_ylabel('Task Complexity', size=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Heatmap saved: {save_path}")
        
        plt.close()
    
    def plot_comparison_bars(self, metrics: Dict, complexity: str, save_path: str = None):
        """
        نمودار میله‌ای مقایسه معیارها
        """
        layers = ['ground', 'edge', 'fog', 'cloud']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Metrics Comparison - {complexity.capitalize()} Tasks',
                    size=16, weight='bold')
        
        # 1. Scalability
        scalability = [metrics.get(l, {}).get('scalability', 0) for l in layers]
        axes[0, 0].bar(layers, scalability, color=[self.colors[l] for l in layers])
        axes[0, 0].set_title('Scalability (%)')
        axes[0, 0].set_ylim(0, 100)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Energy Efficiency
        energy = [metrics.get(l, {}).get('energy_mean', 0) for l in layers]
        axes[0, 1].bar(layers, energy, color=[self.colors[l] for l in layers])
        axes[0, 1].set_title('Average Energy (Joules)')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Throughput
        throughput = [metrics.get(l, {}).get('throughput', 0) for l in layers]
        axes[1, 0].bar(layers, throughput, color=[self.colors[l] for l in layers])
        axes[1, 0].set_title('Throughput (Tasks/s)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Latency
        latency = [metrics.get(l, {}).get('avg_latency', 0) for l in layers]
        axes[1, 1].bar(layers, latency, color=[self.colors[l] for l in layers])
        axes[1, 1].set_title('Average Latency (ms)')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comparison bars saved: {save_path}")
        
        plt.close()
    
    def generate_all_visualizations(self, results_dir: str):
        """
        تولید تمام نمودارها از نتایج ذخیره‌شده
        """
        results_path = Path(results_dir)
        
        complexities = ['simple', 'medium', 'complex']
        all_metrics = {}
        
        # خواندن نتایج
        for comp in complexities:
            metrics_file = results_path / comp / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    all_metrics[comp] = json.load(f)
                
                # نمودارهای هر سطح
                self.plot_radar_chart(
                    all_metrics[comp],
                    comp,
                    str(results_path / comp / "charts" / "radar_chart.png")
                )
                
                self.plot_comparison_bars(
                    all_metrics[comp],
                    comp,
                    str(results_path / comp / "charts" / "comparison_bars.png")
                )
        
        # Heatmap کلی
        if all_metrics:
            self.plot_heatmap(
                all_metrics,
                str(self.output_dir / "scalability_heatmap.png")
            )
        
        print("\n✅ همه نمودارها با موفقیت تولید شدند!")
