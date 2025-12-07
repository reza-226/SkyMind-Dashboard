import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ResultsAnalyzer:
    def __init__(self, results_dir="results/multi_tier_evaluation"):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "final_results.json"
        self.output_dir = self.results_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = {}
        self.df = None
        
        # تنظیم سبک نمودارها
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def load_results(self):
        """بارگذاری نتایج از فایل JSON"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"فایل نتایج یافت نشد: {self.results_file}")
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # استخراج "results" آرایه
        results_list = raw_data.get('results', [])
        
        # تبدیل لیست دیکشنری‌ها به DataFrame
        records = []
        for item in results_list:
            if isinstance(item, dict):
                config = item.get('config', {})
                metrics = item.get('metrics', {})
                training = item.get('training_results', {})
                
                record = {
                    'scenario_id': item.get('scenario_id', 'unknown'),
                    'tier': item.get('tier', 'unknown'),
                    'complexity': item.get('complexity', 'unknown'),
                    'num_tasks': config.get('num_tasks', 0),
                    'num_uavs': config.get('num_uavs', 0),
                    'latency_ms': metrics.get('latency_ms', 0),
                    'energy_joules': metrics.get('energy_joules', 0),
                    'scalability_score': metrics.get('scalability_score', 0),
                    'success_rate': metrics.get('success_rate', 0),
                    'throughput': metrics.get('throughput', 0),
                    'final_reward': training.get('final_reward', 0),
                    'convergence_episode': training.get('convergence_episode', 0),
                    'avg_reward_last_100': training.get('avg_reward_last_100', 0)
                }
                records.append(record)
        
        self.df = pd.DataFrame(records)
        print(f"✅ بارگذاری شد: {len(self.df)} سناریو")
        return self.df
    
    def create_summary_table(self):
        """ایجاد جدول خلاصه"""
        summary = self.df.groupby(['tier', 'complexity']).agg({
            'latency_ms': 'mean',
            'energy_joules': 'mean',
            'success_rate': 'mean',
            'scalability_score': 'mean',
            'throughput': 'mean'
        }).round(3)
        
        # ذخیره CSV
        csv_path = self.output_dir / "summary_table.csv"
        summary.to_csv(csv_path)
        print(f"📊 جدول خلاصه: {csv_path}")
        print("\n### SUMMARY TABLE ###")
        print(summary)
        
        return summary
    
    def plot_latency_heatmap(self):
        """نمودار Heatmap برای Latency"""
        pivot_data = self.df.pivot_table(
            values='latency_ms',
            index='tier',
            columns='complexity',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', cbar_kws={'label': 'Latency (ms)'})
        plt.title('Latency Heatmap by Tier and Complexity')
        plt.xlabel('Complexity')
        plt.ylabel('Tier')
        
        path = self.output_dir / "latency_heatmap.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Heatmap ذخیره شد: {path}")
    
    def plot_energy_latency_tradeoff(self):
        """نمودار Energy vs Latency Tradeoff"""
        plt.figure(figsize=(12, 7))
        
        for tier in self.df['tier'].unique():
            tier_data = self.df[self.df['tier'] == tier]
            plt.scatter(tier_data['latency_ms'], tier_data['energy_joules'], 
                       label=tier, s=100, alpha=0.7)
        
        plt.xlabel('Latency (ms)', fontsize=12)
        plt.ylabel('Energy (Joules)', fontsize=12)
        plt.title('Energy vs Latency Tradeoff', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        path = self.output_dir / "energy_latency_tradeoff.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Tradeoff نمودار ذخیره شد: {path}")
    
    def plot_scalability_comparison(self):
        """نمودار مقایسه Scalability"""
        plt.figure(figsize=(12, 6))
        
        scalability_by_tier = self.df.groupby('tier')['scalability_score'].mean().sort_values(ascending=False)
        colors = plt.cm.viridis(np.linspace(0, 1, len(scalability_by_tier)))
        
        bars = plt.bar(scalability_by_tier.index, scalability_by_tier.values, color=colors)
        plt.ylabel('Scalability Score', fontsize=12)
        plt.xlabel('Tier', fontsize=12)
        plt.title('Scalability Comparison by Tier', fontsize=14)
        
        # اضافه کردن مقادیر بر روی نوارها
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        path = self.output_dir / "scalability_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Scalability نمودار ذخیره شد: {path}")
    
    def plot_success_rate(self):
        """نمودار Success Rate"""
        plt.figure(figsize=(10, 6))
        
        success_rate_by_complexity = self.df.groupby('complexity')['success_rate'].mean().sort_values(ascending=False)
        colors = plt.cm.RdYlGn(success_rate_by_complexity.values / 100)
        
        bars = plt.bar(success_rate_by_complexity.index, success_rate_by_complexity.values, color=colors)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.xlabel('Complexity Level', fontsize=12)
        plt.title('Success Rate by Complexity', fontsize=14)
        plt.ylim(0, 110)
        
        # اضافه کردن درصد بر روی نوارها
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        path = self.output_dir / "success_rate.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Success Rate نمودار ذخیره شد: {path}")
    
    def generate_all_visualizations(self):
        """تولید تمام نمودارها و تحلیل‌ها"""
        print("\n" + "="*60)
        print("📊 شروع تولید نمودارها و تحلیل‌ها")
        print("="*60 + "\n")
        
        self.load_results()
        self.create_summary_table()
        self.plot_latency_heatmap()
        self.plot_energy_latency_tradeoff()
        self.plot_scalability_comparison()
        self.plot_success_rate()
        
        print("\n" + "="*60)
        print("✅ تمام تحلیل‌ها و نمودارها ایجاد شد")
        print(f"📁 خروجی در: {self.output_dir}")
        print("="*60 + "\n")

if __name__ == "__main__":
    analyzer = ResultsAnalyzer()
    analyzer.generate_all_visualizations()
