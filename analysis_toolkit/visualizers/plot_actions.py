"""
رسم نمودارهای مربوط به اکشن‌ها
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns


class ActionPlotter:
    """رسم نمودارهای اکشن‌ها"""
    
    def __init__(self, output_dir: str = 'plots'):
        """
        Args:
            output_dir: پوشه ذخیره نمودارها
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_offload_distribution(self,
                                 offload_stats: Dict,
                                 save_name: Optional[str] = None):
        """
        رسم توزیع انتخاب‌های offload
        
        Args:
            offload_stats: آمار offload از ActionAnalyzer
            save_name: نام فایل برای ذخیره
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        locations = list(offload_stats['percentages'].keys())
        percentages = list(offload_stats['percentages'].values())
        counts = [offload_stats['counts'][loc] for loc in locations]
        
        colors = sns.color_palette("husl", len(locations))
        
        # نمودار دایره‌ای
        wedges, texts, autotexts = ax1.pie(percentages, labels=locations, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
        ax1.set_title('Offload Location Distribution', fontsize=14, fontweight='bold')
        
        # نمودار میله‌ای
        ax2.bar(locations, counts, color=colors, alpha=0.7)
        ax2.set_xlabel('Location', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Offload Selection Frequency', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        
        plt.show()
    
    def plot_resource_allocation(self,
                                cpu_stats: Dict,
                                bandwidth_stats: Dict,
                                save_name: Optional[str] = None):
        """
        رسم نمودارهای تخصیص منابع
        
        Args:
            cpu_stats: آمار CPU
            bandwidth_stats: آمار Bandwidth
            save_name: نام فایل برای ذخیره
        """
        fig = plt.figure(figsize=(16, 6))
        
        # نمودار 1: CPU Allocation
        ax1 = plt.subplot(1, 3, 1)
        cpu_data = [cpu_stats['min'], cpu_stats['mean'], cpu_stats['max']]
        labels = ['Min', 'Mean', 'Max']
        colors_cpu = ['lightcoral', 'skyblue', 'lightgreen']
        
        bars = ax1.bar(labels, cpu_data, color=colors_cpu, alpha=0.7)
        ax1.set_ylabel('CPU Allocation', fontsize=12)
        ax1.set_title('CPU Allocation Statistics', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # اضافه کردن مقادیر روی میله‌ها
        for bar, value in zip(bars, cpu_data):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # نمودار 2: Bandwidth Distribution (Mean)
        ax2 = plt.subplot(1, 3, 2)
        channels = [f"CH{i}" for i in range(len(bandwidth_stats))]
        bw_means = [bandwidth_stats[f'channel_{i}']['mean'] for i in range(len(bandwidth_stats))]
        bw_stds = [bandwidth_stats[f'channel_{i}']['std'] for i in range(len(bandwidth_stats))]
        
        colors_bw = sns.color_palette("Set2", len(channels))
        bars = ax2.bar(channels, bw_means, yerr=bw_stds, color=colors_bw, 
                      alpha=0.7, capsize=5)
        ax2.set_ylabel('Bandwidth Allocation', fontsize=12)
        ax2.set_title('Bandwidth Distribution (Mean ± Std)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # نمودار 3: Bandwidth Box Plot
        ax3 = plt.subplot(1, 3, 3)
        bw_data = [[bandwidth_stats[f'channel_{i}']['min'],
                   bandwidth_stats[f'channel_{i}']['mean'],
                   bandwidth_stats[f'channel_{i}']['max']] 
                  for i in range(len(bandwidth_stats))]
        
        bp = ax3.boxplot(bw_data, labels=channels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_bw):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Bandwidth Allocation', fontsize=12)
        ax3.set_title('Bandwidth Range Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        
        plt.show()
    
    def plot_movement_pattern(self,
                            movement_stats: Dict,
                            save_name: Optional[str] = None):
        """
        رسم الگوی حرکت
        
        Args:
            movement_stats: آمار حرکت
            save_name: نام فایل برای ذخیره
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # نمودار 1: X Movement Statistics
        ax1 = axes[0, 0]
        x_data = [movement_stats['x']['min'], movement_stats['x']['mean'], 
                 movement_stats['x']['max']]
        labels = ['Min', 'Mean', 'Max']
        ax1.bar(labels, x_data, color=['lightcoral', 'skyblue', 'lightgreen'], alpha=0.7)
        ax1.set_ylabel('X Displacement', fontsize=12)
        ax1.set_title('X-Axis Movement Statistics', fontsize=13, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # نمودار 2: Y Movement Statistics
        ax2 = axes[0, 1]
        y_data = [movement_stats['y']['min'], movement_stats['y']['mean'], 
                 movement_stats['y']['max']]
        ax2.bar(labels, y_data, color=['lightcoral', 'skyblue', 'lightgreen'], alpha=0.7)
        ax2.set_ylabel('Y Displacement', fontsize=12)
        ax2.set_title('Y-Axis Movement Statistics', fontsize=13, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # نمودار 3: Total Distance
        ax3 = axes[1, 0]
        dist_labels = ['Mean\nDistance', 'Total\nDistance']
        dist_data = [movement_stats['total_distance']['mean'],
                    movement_stats['total_distance']['sum']]
        ax3.bar(dist_labels, dist_data, color=['orange', 'purple'], alpha=0.7)
        ax3.set_ylabel('Distance', fontsize=12)
        ax3.set_title('Movement Distance Statistics', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # نمودار 4: Movement Summary Table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_data = [
            ['Metric', 'X-Axis', 'Y-Axis'],
            ['Mean', f"{movement_stats['x']['mean']:.2f}", f"{movement_stats['y']['mean']:.2f}"],
            ['Std', f"{movement_stats['x']['std']:.2f}", f"{movement_stats['y']['std']:.2f}"],
            ['Min', f"{movement_stats['x']['min']:.2f}", f"{movement_stats['y']['min']:.2f}"],
            ['Max', f"{movement_stats['x']['max']:.2f}", f"{movement_stats['y']['max']:.2f}"]
        ]
        
        table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # رنگ‌آمیزی سطر اول
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Movement Statistics Summary', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        
        plt.show()
