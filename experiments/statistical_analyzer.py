import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class NumpyEncoder(json.JSONEncoder):
    """JSON Encoder برای تبدیل انواع داده NumPy"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class StatisticalAnalyzer:
    def __init__(self, results_file='results/multi_tier_evaluation/final_results.json'):
        self.results_file = Path(results_file)
        self.output_dir = Path('results/multi_tier_evaluation/statistical_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # بارگذاری داده‌ها
        self.data = {
            'tier': [],
            'complexity': [],
            'latency': [],
            'energy': [],
            'success_rate': [],
            'scalability': [],
            'throughput': []
        }
        self.df = None
        
    def load_results(self):
        """بارگذاری نتایج از فایل JSON"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # استخراج آرایه results
        scenarios = data.get('results', [])
        
        for scenario_data in scenarios:
            # ✅ استخراج config و metrics
            config = scenario_data.get('config', {})
            metrics = scenario_data.get('metrics', {})
            
            # بررسی وجود metrics
            if not metrics:
                print(f"⚠️ هشدار: metrics برای {scenario_data.get('scenario_id')} یافت نشد!")
                continue
            
            tier = scenario_data.get('tier')
            complexity = scenario_data.get('complexity')
            
            # ذخیره داده‌ها
            self.data['tier'].append(tier)
            self.data['complexity'].append(complexity)
            self.data['latency'].append(metrics.get('latency_ms', 0))
            self.data['energy'].append(metrics.get('energy_joules', 0))
            self.data['success_rate'].append(metrics.get('success_rate', 0))
            self.data['scalability'].append(metrics.get('scalability_score', 0))
            self.data['throughput'].append(metrics.get('throughput', 0))
        
        self.df = pd.DataFrame(self.data)
        print(f"✅ بارگذاری شد: {len(self.df)} سناریو\n")
        
    def descriptive_statistics(self):
        """محاسبه آمار توصیفی"""
        print("\n" + "="*70)
        print("📊 آمار توصیفی (Descriptive Statistics)")
        print("="*70)
        
        # آمار کلی
        desc = self.df.describe()
        print("\n1️⃣ خلاصه آماری کلی:")
        print(desc.to_string())
        
        # آمار به تفکیک Tier
        print("\n2️⃣ میانگین متریک‌ها به تفکیک Tier:")
        tier_stats = self.df.groupby('tier')[['latency', 'energy', 'success_rate', 'throughput']].mean()
        print(tier_stats.to_string())
        
        # آمار به تفکیک Complexity
        print("\n3️⃣ میانگین متریک‌ها به تفکیک Complexity:")
        complexity_stats = self.df.groupby('complexity')[['latency', 'energy', 'success_rate', 'throughput']].mean()
        print(complexity_stats.to_string())
        
        # ذخیره
        with open(self.output_dir / 'descriptive_stats.txt', 'w', encoding='utf-8') as f:
            f.write("DESCRIPTIVE STATISTICS\n")
            f.write("="*70 + "\n\n")
            f.write("Overall Summary:\n")
            f.write(desc.to_string() + "\n\n")
            f.write("By Tier:\n")
            f.write(tier_stats.to_string() + "\n\n")
            f.write("By Complexity:\n")
            f.write(complexity_stats.to_string())
            
    def anova_analysis(self):
        """آنالیز واریانس (ANOVA)"""
        print("\n" + "="*70)
        print("🔬 تحلیل واریانس - ANOVA")
        print("="*70)
        
        results = {}
        metrics = ['latency', 'energy', 'success_rate', 'throughput']
        
        for metric in metrics:
            # ANOVA برای Tier
            groups_tier = [self.df[self.df['tier'] == tier][metric].values 
                          for tier in self.df['tier'].unique()]
            f_stat_tier, p_value_tier = stats.f_oneway(*groups_tier)
            
            # ANOVA برای Complexity
            groups_complexity = [self.df[self.df['complexity'] == comp][metric].values 
                               for comp in self.df['complexity'].unique()]
            f_stat_comp, p_value_comp = stats.f_oneway(*groups_complexity)
            
            results[metric] = {
                'tier': {'F-statistic': float(f_stat_tier), 'p-value': float(p_value_tier)},
                'complexity': {'F-statistic': float(f_stat_comp), 'p-value': float(p_value_comp)}
            }
            
            print(f"\n📌 {metric.upper()}:")
            print(f"   Tier: F={f_stat_tier:.4f}, p={p_value_tier:.4e} {'✅ معنادار' if p_value_tier < 0.05 else '❌ غیرمعنادار'}")
            print(f"   Complexity: F={f_stat_comp:.4f}, p={p_value_comp:.4e} {'✅ معنادار' if p_value_comp < 0.05 else '❌ غیرمعنادار'}")
        
        # ذخیره با NumpyEncoder
        with open(self.output_dir / 'anova_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
            
    def pairwise_ttest(self):
        """تست t زوجی (Pairwise T-Test) با Cohen's d"""
        print("\n" + "="*70)
        print("📊 تست t زوجی (Pairwise T-Test with Cohen's d)")
        print("="*70)
        
        tiers = self.df['tier'].unique()
        metrics = ['latency', 'energy', 'success_rate', 'throughput']
        
        results = {}
        
        for metric in metrics:
            print(f"\n🔹 {metric.upper()}:")
            results[metric] = {}
            
            for i, tier1 in enumerate(tiers):
                for tier2 in tiers[i+1:]:
                    data1 = self.df[self.df['tier'] == tier1][metric]
                    data2 = self.df[self.df['tier'] == tier2][metric]
                    
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    
                    # Cohen's d
                    pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
                    cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    comparison = f"{tier1} vs {tier2}"
                    
                    # ✅ تبدیل صریح به انواع پایتون
                    results[metric][comparison] = {
                        't-statistic': float(t_stat),
                        'p-value': float(p_value),
                        'cohens_d': float(cohens_d),
                        'significant': bool(p_value < 0.05)  # ← اصلاح شد
                    }
                    
                    print(f"   {comparison}: t={t_stat:.4f}, p={p_value:.4e}, d={cohens_d:.4f} "
                          f"{'✅ معنادار' if p_value < 0.05 else '❌'}")
        
        # ذخیره با NumpyEncoder
        with open(self.output_dir / 'pairwise_ttest.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
            
    def correlation_analysis(self):
        """آنالیز همبستگی (Pearson & Spearman)"""
        print("\n" + "="*70)
        print("🔗 تحلیل همبستگی (Correlation Analysis)")
        print("="*70)
        
        metrics = ['latency', 'energy', 'success_rate', 'scalability', 'throughput']
        
        # Pearson
        pearson_corr = self.df[metrics].corr(method='pearson')
        print("\n📌 Pearson Correlation:")
        print(pearson_corr.to_string())
        
        # Spearman
        spearman_corr = self.df[metrics].corr(method='spearman')
        print("\n📌 Spearman Correlation:")
        print(spearman_corr.to_string())
        
        # نمودار
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(pearson_corr, annot=True, fmt='.3f', cmap='coolwarm', ax=axes[0], vmin=-1, vmax=1)
        axes[0].set_title('Pearson Correlation')
        
        sns.heatmap(spearman_corr, annot=True, fmt='.3f', cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1)
        axes[1].set_title('Spearman Correlation')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 ذخیره شد: correlation_matrix.png")
        
        # ذخیره CSV
        pearson_corr.to_csv(self.output_dir / 'pearson_correlation.csv')
        spearman_corr.to_csv(self.output_dir / 'spearman_correlation.csv')
        
    def regression_analysis(self):
        """رگرسیون خطی (Linear Regression)"""
        print("\n" + "="*70)
        print("📈 تحلیل رگرسیون (Regression Analysis)")
        print("="*70)
        
        # Latency vs Energy
        X = self.df[['latency']].values
        y = self.df['energy'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        r2 = model.score(X, y)
        coef = model.coef_[0]
        intercept = model.intercept_
        
        print(f"\n🔹 Latency → Energy:")
        print(f"   R² = {r2:.4f}")
        print(f"   معادله: Energy = {coef:.6f} × Latency + {intercept:.6f}")
        
        # نمودار
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['latency'], self.df['energy'], alpha=0.6, label='Data Points')
        plt.plot(X, model.predict(X), color='red', linewidth=2, label=f'Regression Line (R²={r2:.3f})')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Energy (J)')
        plt.title('Regression: Latency vs Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'regression_latency_energy.png', dpi=300, bbox_inches='tight')
        print(f"💾 ذخیره شد: regression_latency_energy.png")
        
    def generate_report(self):
        """تولید گزارش نهایی"""
        print("\n" + "="*70)
        print("📄 تولید گزارش نهایی")
        print("="*70)
        
        report = []
        report.append("="*70)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("="*70)
        report.append(f"\nتعداد کل سناریوها: {len(self.df)}")
        report.append(f"Tiers: {', '.join(self.df['tier'].unique())}")
        report.append(f"Complexity Levels: {', '.join(self.df['complexity'].unique())}")
        
        report.append("\n\n1️⃣ یافته‌های کلیدی:")
        report.append(f"   • بهترین Tier از نظر Latency: {self.df.groupby('tier')['latency'].mean().idxmin()}")
        report.append(f"   • بهترین Tier از نظر Energy: {self.df.groupby('tier')['energy'].mean().idxmin()}")
        report.append(f"   • بهترین Tier از نظر Success Rate: {self.df.groupby('tier')['success_rate'].mean().idxmax()}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open(self.output_dir / 'final_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        print(f"\n✅ گزارش نهایی ذخیره شد: {self.output_dir / 'final_report.txt'}")
        
    def run_all_analyses(self):
        """اجرای تمام تحلیل‌ها"""
        print("\n" + "="*70)
        print("🚀 شروع تحلیل آماری جامع")
        print("="*70)
        
        self.load_results()
        self.descriptive_statistics()
        self.anova_analysis()
        self.pairwise_ttest()
        self.correlation_analysis()
        self.regression_analysis()
        self.generate_report()
        
        print("\n" + "="*70)
        print("✅ تحلیل آماری با موفقیت تکمیل شد!")
        print(f"📂 فایل‌های خروجی: {self.output_dir}")
        print("="*70)

if __name__ == "__main__":
    analyzer = StatisticalAnalyzer()
    analyzer.run_all_analyses()
