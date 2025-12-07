"""
رسم نمودارهای مربوط به reward
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns


class RewardPlotter:
    """رسم نمودارهای reward"""
    
    def __init__(self, output_dir: str = 'plots'):
        """
        Args:
            output_dir: پوشه ذخیره نمودارها
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # تنظیمات استایل
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
    
    def plot_training_curve(self, 
                           episodes: List[int], 
                           rewards: List[float],
                           title: str = "Training Progress",
                           window: int = 50,
                           save_name: Optional[str] = None):
        """
        رسم منحنی آموزش با میانگین متحرک
        
        Args:
            episodes: لیست شماره اپیزودها
            rewards: لیست reward‌ها
            title: عنوان نمودار
            window: پنجره برای میانگین متحرک
            save_name: نام فایل برای ذخیره (اختیاری)
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # رسم reward‌های خام (با شفافیت)
        ax.plot(episodes, rewards, alpha=0.3, color='gray', label='Raw Rewards')
        
        # محاسبه و رسم میانگین متحرک
        if len(rewards) >= window:
            moving_avg = self._moving_average(rewards, window)
            ax.plot(episodes[window-1:], moving_avg, 
                   color='blue', linewidth=2, label=f'Moving Average (window={window})')
        
        # محاسبه و رسم بهترین reward تا کنون
        best_so_far = np.maximum.accumulate(rewards)
        ax.plot(episodes, best_so_far, 
               color='green', linewidth=2, linestyle='--', label='Best So Far')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        
        plt.show()
    
    def plot_comparison(self,
                       model_results: Dict[str, Dict],
                       save_name: Optional[str] = None):
        """
        مقایسه چند مدل
        
        Args:
            model_results: دیکشنری نتایج مدل‌ها
            save_name: نام فایل برای ذخیره
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # آماده‌سازی داده‌ها
        model_names = list(model_results.keys())
        means = [res['statistics']['mean_reward'] for res in model_results.values()]
        stds = [res['statistics']['std_reward'] for res in model_results.values()]
        
        # نمودار 1: میانگین با خطا
        colors = sns.color_palette("husl", len(model_names))
        x_pos = np.arange(len(model_names))
        
        ax1.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Mean Reward', fontsize=12)
        ax1.set_title('Model Comparison - Mean Rewards', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # نمودار 2: Box plot
        all_rewards = [res['all_rewards'] for res in model_results.values()]
        bp = ax2.boxplot(all_rewards, labels=model_names, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Reward Distribution', fontsize=12)
        ax2.set_title('Model Comparison - Reward Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        
        plt.show()
    
    def plot_reward_distribution(self,
                                rewards: List[float],
                                title: str = "Reward Distribution",
                                save_name: Optional[str] = None):
        """
        رسم توزیع reward‌ها
        
        Args:
            rewards: لیست reward‌ها
            title: عنوان نمودار
            save_name: نام فایل برای ذخیره
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # هیستوگرام
        ax1.hist(rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(rewards), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        ax1.axvline(np.median(rewards), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(rewards):.2f}')
        ax1.set_xlabel('Reward', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'{title} - Histogram', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Box plot
        bp = ax2.boxplot(rewards, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][0].set_alpha(0.7)
        
        ax2.set_ylabel('Reward', fontsize=12)
        ax2.set_title(f'{title} - Box Plot', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # اضافه کردن آمار
        stats_text = f"Mean: {np.mean(rewards):.2f}\n"
        stats_text += f"Std: {np.std(rewards):.2f}\n"
        stats_text += f"Min: {np.min(rewards):.2f}\n"
        stats_text += f"Max: {np.max(rewards):.2f}"
        
        ax2.text(1.15, 0.5, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / f"{save_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        
        plt.show()
    
    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """محاسبه میانگین متحرک"""
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='valid')
