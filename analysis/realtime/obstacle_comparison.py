"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 تحلیل مقایسه‌ای موانع
مسیر: analysis/realtime/obstacle_comparison.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json

class ObstacleComparison:
    """کلاس مقایسه عملکرد در سطوح مختلف موانع"""
    
    def __init__(self):
        self.results = {
            'simple': {},
            'medium': {},
            'complex': {}
        }
        
        self.metrics = [
            'avg_delay',
            'avg_energy',
            'success_rate',
            'collision_rate',
            'path_length',
            'computation_time'
        ]
        
        self.algorithms = ['MADDPG', 'DQN', 'BLS', 'GA', 'ECORI']
        self.layers = ['Ground', 'Local', 'Edge', 'Cloud']
    
    def add_result(self, 
                   complexity: str, 
                   algorithm: str, 
                   layer: str, 
                   metrics: Dict):
        """
        اضافه کردن نتیجه یک آزمایش
        
        Args:
            complexity: 'simple', 'medium', 'complex'
            algorithm: نام الگوریتم
            layer: نام لایه
            metrics: دیکشنری متریک‌ها
        """
        key = f"{algorithm}_{layer}"
        self.results[complexity][key] = metrics
    
    def generate_intra_complexity_comparison(self, complexity: str):
        """
        📈 مقایسه داخلی: الگوریتم‌ها در یک سطح پیچیدگی
        """
        data = []
        
        for algo in self.algorithms:
            for layer in self.layers:
                key = f"{algo}_{layer}"
                if key in self.results[complexity]:
                    metrics = self.results[complexity][key]
                    data.append({
                        'Algorithm': algo,
                        'Layer': layer,
                        **metrics
                    })
        
        df = pd.DataFrame(data)
        
        # ایجاد نمودارها
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Intra-Complexity Comparison: {complexity.upper()}',
                    fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('avg_delay', 'Average Delay (ms)'),
            ('avg_energy', 'Average Energy (J)'),
            ('success_rate', 'Success Rate (%)'),
            ('collision_rate', 'Collision Rate (%)'),
            ('path_length', 'Average Path Length (m)'),
            ('computation_time', 'Computation Time (s)')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            # Bar plot
            pivot = df.pivot(index='Algorithm', columns='Layer', values=metric)
            pivot.plot(kind='bar', ax=ax, width=0.8)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Algorithm', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.legend(title='Layer', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            # چرخش برچسب‌ها
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'results/intra_comparison_{complexity}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return df
    
    def generate_inter_layer_comparison(self, complexity: str, algorithm: str):
        """
        📊 مقایسه بین‌لایه‌ای: لایه‌ها برای یک الگوریتم
        """
        data = []
        
        for layer in self.layers:
            key = f"{algorithm}_{layer}"
            if key in self.results[complexity]:
                metrics = self.results[complexity][key]
                data.append({
                    'Layer': layer,
                    **metrics
                })
        
        df = pd.DataFrame(data)
        
        # نمودار رادار
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # نرمال‌سازی متریک‌ها
        normalized_metrics = ['avg_delay_norm', 'avg_energy_norm', 
                            'success_rate', 'collision_rate_inv']
        
        angles = np.linspace(0, 2 * np.pi, len(normalized_metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for _, row in df.iterrows():
            values = [
                1 - row['avg_delay'] / df['avg_delay'].max(),
                1 - row['avg_energy'] / df['avg_energy'].max(),
                row['success_rate'],
                1 - row['collision_rate']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Layer'])
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Delay↓', 'Energy↓', 'Success↑', 'Safety↑'])
        ax.set_ylim(0, 1)
        ax.set_title(f'Layer Comparison: {algorithm} ({complexity})',
                    size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/inter_layer_{algorithm}_{complexity}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return df
    
    def generate_cross_complexity_comparison(self, algorithm: str, layer: str):
        """
        🔄 مقایسه متقاطع: تأثیر افزایش پیچیدگی
        """
        data = []
        
        for complexity in ['simple', 'medium', 'complex']:
            key = f"{algorithm}_{layer}"
            if key in self.results[complexity]:
                metrics = self.results[complexity][key]
                data.append({
                    'Complexity': complexity,
                    **metrics
                })
        
        df = pd.DataFrame(data)
        
        # نمودار خطی
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Cross-Complexity Analysis: {algorithm} on {layer}',
                    fontsize=14, fontweight='bold')
        
        metrics_plot = [
            ('avg_delay', 'Delay (ms)', axes[0, 0]),
            ('avg_energy', 'Energy (J)', axes[0, 1]),
            ('success_rate', 'Success Rate (%)', axes[1, 0]),
            ('collision_rate', 'Collision Rate (%)', axes[1, 1])
        ]
        
        for metric, ylabel, ax in metrics_plot:
            ax.plot(df['Complexity'], df[metric], 
                   marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Complexity Level', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # اضافه کردن مقادیر روی نقاط
            for x, y in zip(df['Complexity'], df[metric]):
                ax.annotate(f'{y:.2f}', xy=(x, y), 
                          textcoords='offset points', xytext=(0, 10),
                          ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'results/cross_complexity_{algorithm}_{layer}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return df
    
    def generate_heatmap_comparison(self):
        """
        🌡️ نمودار حرارتی: الگوریتم × سطح پیچیدگی
        """
        # متریک: Average Delay
        data = []
        
        for complexity in ['simple', 'medium', 'complex']:
            row = []
            for algo in self.algorithms:
                # میانگین روی همه لایه‌ها
                values = []
                for layer in self.layers:
                    key = f"{algo}_{layer}"
                    if key in self.results[complexity]:
                        values.append(self.results[complexity][key]['avg_delay'])
                
                row.append(np.mean(values) if values else np.nan)
            
            data.append(row)
        
        df = pd.DataFrame(data, 
                         index=['Simple', 'Medium', 'Complex'],
                         columns=self.algorithms)
        
        # رسم heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Average Delay (ms)'})
        plt.title('Algorithm Performance Across Complexity Levels\n(Lower is Better)',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Complexity Level', fontsize=12)
        plt.tight_layout()
        plt.savefig('results/heatmap_complexity_algo.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_table(self):
        """
        📋 جدول خلاصه نتایج
        """
        rows = []
        
        for complexity in ['simple', 'medium', 'complex']:
            for algo in self.algorithms:
                for layer in self.layers:
                    key = f"{algo}_{layer}"
                    if key in self.results[complexity]:
                        metrics = self.results[complexity][key]
                        rows.append({
                            'Complexity': complexity.capitalize(),
                            'Algorithm': algo,
                            'Layer': layer,
                            'Delay (ms)': f"{metrics['avg_delay']:.2f}",
                            'Energy (J)': f"{metrics['avg_energy']:.2f}",
                            'Success (%)': f"{metrics['success_rate']:.1f}",
                            'Collision (%)': f"{metrics['collision_rate']:.1f}"
                        })
        
        df = pd.DataFrame(rows)
        
        # ذخیره به CSV
        df.to_csv('results/obstacle_comparison_summary.csv', index=False)
        
        # ذخیره به LaTeX
        latex_table = df.to_latex(index=False, 
                                  caption='Performance comparison across obstacle complexities',
                                  label='tab:obstacle_comparison')
        
        with open('results/obstacle_comparison_table.tex', 'w') as f:
            f.write(latex_table)
        
        return df
    
    def save_results(self, filename: str = 'obstacle_comparison_results.json'):
        """ذخیره نتایج به فایل JSON"""
        with open(f'results/{filename}', 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filename: str = 'obstacle_comparison_results.json'):
        """بارگذاری نتایج از فایل JSON"""
        with open(f'results/{filename}', 'r') as f:
            self.results = json.load(f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧪 مثال استفاده
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # ایجاد نمونه
    comparison = ObstacleComparison()
    
    # فرض: داده‌های نمونه (در واقعیت از شبیه‌سازی می‌آید)
    np.random.seed(42)
    
    for complexity in ['simple', 'medium', 'complex']:
        # ضریب سختی
        difficulty_factor = {'simple': 1.0, 'medium': 1.5, 'complex': 2.0}[complexity]
        
        for algo in ['MADDPG', 'DQN', 'BLS', 'GA']:
            # فرض: MADDPG بهتر عمل می‌کند
            algo_factor = {'MADDPG': 0.8, 'DQN': 1.0, 'BLS': 1.3, 'GA': 1.5}[algo]
            
            for layer in ['Ground', 'Local', 'Edge', 'Cloud']:
                # فرض: Edge بهینه است
                layer_factor = {'Ground': 1.2, 'Local': 1.1, 'Edge': 0.9, 'Cloud': 1.0}[layer]
                
                metrics = {
                    'avg_delay': np.random.uniform(50, 150) * difficulty_factor * algo_factor * layer_factor,
                    'avg_energy': np.random.uniform(10, 50) * difficulty_factor * algo_factor,
                    'success_rate': max(60, 100 - np.random.uniform(5, 20) * difficulty_factor * algo_factor),
                    'collision_rate': min(30, np.random.uniform(1, 10) * difficulty_factor / algo_factor),
                    'path_length': np.random.uniform(200, 500) * difficulty_factor,
                    'computation_time': np.random.uniform(0.5, 3) * difficulty_factor * algo_factor
                }
                
                comparison.add_result(complexity, algo, layer, metrics)
    
    # تولید تحلیل‌ها
    print("🔄 در حال تولید تحلیل‌های مقایسه‌ای...")
    
    # مقایسه داخلی
    for complexity in ['simple', 'medium', 'complex']:
        df = comparison.generate_intra_complexity_comparison(complexity)
        print(f"✅ مقایسه داخلی {complexity} ذخیره شد")
    
    # مقایسه بین‌لایه‌ای
    for algo in ['MADDPG', 'DQN']:
        for complexity in ['simple', 'complex']:
            df = comparison.generate_inter_layer_comparison(complexity, algo)
            print(f"✅ مقایسه لایه‌ها {algo} در {complexity} ذخیره شد")
    
    # مقایسه متقاطع
    for algo in ['MADDPG', 'BLS']:
        for layer in ['Edge', 'Cloud']:
            df = comparison.generate_cross_complexity_comparison(algo, layer)
            print(f"✅ تحلیل متقاطع {algo} روی {layer} ذخیره شد")
    
    # Heatmap
    comparison.generate_heatmap_comparison()
    print("✅ Heatmap تولید شد")
    
    # جدول خلاصه
    summary_df = comparison.generate_summary_table()
    print("✅ جدول خلاصه ذخیره شد")
    
    # ذخیره نتایج
    comparison.save_results()
    print("✅ نتایج JSON ذخیره شد")
    
    print("\n" + "━" * 60)
    print("🎉 تمام تحلیل‌های مقایسه‌ای با موفقیت تولید شدند!")
    print("📁 مسیر: results/")
    print("━" * 60)
