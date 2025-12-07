#!/usr/bin/env python3
"""
Training Status Analysis Script - COMPLETE FIX
تحلیل وضعیت آموزش با پشتیبانی کامل از جستجوی recursive
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# تنظیمات نمایش
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'

class TrainingAnalyzer:
    """تحلیلگر وضعیت آموزش"""
    
    def __init__(self, results_dir='models'):
        self.results_dir = Path(results_dir)
        self.viz_dir = Path('visualizations')
        self.viz_dir.mkdir(exist_ok=True)
        
        # شناسایی خودکار فایل‌ها
        self.auto_detect_files()
        
    def auto_detect_files(self):
        """شناسایی خودکار مسیر فایل‌های training_results.json با جستجوی recursive"""
        
        print("\n🔍 جستجوی فایل‌های Training Results...\n")
        
        # جستجوی recursive با Path.rglob
        found_files = []
        
        # جستجو از root directory
        for pattern in ['models', 'results', '.']:
            search_path = Path(pattern)
            if search_path.exists():
                matches = list(search_path.rglob('training_results.json'))
                found_files.extend([str(f) for f in matches])
        
        # حذف تکراری‌ها و مرتب‌سازی
        found_files = sorted(list(set(found_files)))
        
        if found_files:
            print(f"✅ {len(found_files)} فایل پیدا شد:")
            for f in found_files:
                size_mb = os.path.getsize(f) / 1024 / 1024
                print(f"   📄 {f} ({size_mb:.2f} MB)")
        else:
            print("⚠️  هیچ فایل training_results.json پیدا نشد!")
            print("\n💡 راهنمایی:")
            print("   1. اطمینان حاصل کنید فایل‌ها با نام training_results.json موجود هستند")
            print("   2. یا مسیر کامل فایل‌ها رو بررسی کنید")
        
        self.found_files = found_files
        print()
    
    def load_all_training_files(self):
        """بارگذاری تمام فایل‌های training results"""
        
        results_data = {}
        
        for file_path in self.found_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                level_name = data.get('level_name', Path(file_path).parent.name)
                results_data[level_name] = data
                
                total_eps = data['results']['total_episodes']
                print(f"✅ بارگذاری {level_name}: {total_eps} اپیزود")
                
            except Exception as e:
                print(f"❌ خطا در بارگذاری {file_path}: {e}")
        
        return results_data
    
    def analyze_convergence(self, rewards, window=100):
        """بررسی همگرایی"""
        
        if len(rewards) < window:
            return {
                'status': 'insufficient_data',
                'converged': False,
                'message': f'⚠️ تعداد اپیزودها ({len(rewards)}) کمتر از window ({window}) است'
            }
        
        # محاسبه آمارها
        last_100 = rewards[-100:]
        prev_100 = rewards[-200:-100] if len(rewards) >= 200 else rewards[:100]
        
        mean_last_100 = np.mean(last_100)
        std_last_100 = np.std(last_100)
        mean_prev_100 = np.mean(prev_100)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)
        
        improvement = mean_last_100 - mean_prev_100
        improvement_pct = (improvement / abs(mean_prev_100) * 100) if mean_prev_100 != 0 else 0
        
        # معیارهای همگرایی
        is_improving = improvement > 0
        is_stable = std_last_100 < 10
        is_converged = is_stable and abs(improvement_pct) < 5
        
        # تعیین وضعیت با توجه به مقادیر منفی
        if mean_last_100 < -20:
            status = 'poor_performance'
            message = '🔴 عملکرد ضعیف - نیاز به آموزش بیشتر'
        elif is_converged:
            status = 'converged'
            message = '✅ مدل به همگرایی رسیده'
        elif is_improving:
            status = 'improving'
            message = '📈 مدل در حال بهبود است'
        else:
            status = 'stuck'
            message = '⚠️ مدل stuck شده'
        
        return {
            'status': status,
            'converged': is_converged,
            'message': message,
            'metrics': {
                'mean_last_100': mean_last_100,
                'std_last_100': std_last_100,
                'mean_prev_100': mean_prev_100,
                'max_reward': max_reward,
                'min_reward': min_reward,
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'is_stable': is_stable,
                'is_improving': is_improving,
                'total_episodes': len(rewards)
            }
        }
    
    def plot_level_analysis(self, level_name, data):
        """رسم نمودار تحلیل یک سطح"""
        
        history = data['results']['training_history']
        episodes = history['episodes']
        rewards = history['rewards']
        noise = history.get('noise_std', [])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Analysis - {level_name}', fontsize=16, fontweight='bold')
        
        # 1. Raw Rewards
        axes[0, 0].plot(episodes, rewards, alpha=0.5, linewidth=0.5, color='blue')
        axes[0, 0].set_title('Raw Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.3)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Moving Average
        window = 50
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(ma, linewidth=2, color='red')
            axes[0, 1].set_title(f'Moving Average (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Avg Reward')
            axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Best Reward Progress
        best_rewards = np.maximum.accumulate(rewards)
        axes[0, 2].plot(episodes, best_rewards, linewidth=2, color='green')
        axes[0, 2].set_title('Best Reward Progress')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Best Reward')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Reward Distribution
        axes[1, 0].hist(rewards, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[1, 0].axvline(np.mean(rewards), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        axes[1, 0].axvline(np.median(rewards), color='orange', linestyle='--',
                          linewidth=2, label=f'Median: {np.median(rewards):.2f}')
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Noise Decay
        if noise:
            axes[1, 1].plot(episodes, noise, linewidth=2, color='purple')
            axes[1, 1].set_title('Exploration Noise Decay')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Noise Std')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Noise Data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=14)
            axes[1, 1].set_title('Exploration Noise')
        
        # 6. Recent Performance (Last 100)
        last_100_eps = episodes[-100:] if len(episodes) >= 100 else episodes
        last_100_rew = rewards[-100:] if len(rewards) >= 100 else rewards
        
        axes[1, 2].plot(last_100_eps, last_100_rew, linewidth=2, color='orange')
        axes[1, 2].axhline(np.mean(last_100_rew), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(last_100_rew):.2f}')
        axes[1, 2].set_title(f'Last {len(last_100_rew)} Episodes')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # ذخیره
        output_path = self.viz_dir / f'training_analysis_{level_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ نمودار ذخیره شد: {output_path}")
        
        plt.close()
    
    def generate_report(self, results_dict):
        """تولید گزارش متنی جامع"""
        
        report_path = self.viz_dir / 'convergence_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING CONVERGENCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # خلاصه کلی
            total_converged = sum(1 for r in results_dict.values() if r.get('converged', False))
            total_levels = len(results_dict)
            
            f.write(f"📊 SUMMARY:\n")
            f.write(f"   Total Levels: {total_levels}\n")
            f.write(f"   Converged: {total_converged}\n")
            f.write(f"   In Progress: {total_levels - total_converged}\n\n")
            
            # جزئیات هر سطح
            for level_name, analysis in results_dict.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"Level: {level_name}\n")
                f.write(f"{'=' * 80}\n\n")
                
                f.write(f"Status: {analysis['status']}\n")
                f.write(f"Message: {analysis['message']}\n\n")
                
                if 'metrics' in analysis:
                    m = analysis['metrics']
                    f.write("📈 Metrics:\n")
                    f.write(f"   Total Episodes:     {m['total_episodes']}\n")
                    f.write(f"   Mean (Last 100):    {m['mean_last_100']:.4f}\n")
                    f.write(f"   Std (Last 100):     {m['std_last_100']:.4f}\n")
                    f.write(f"   Mean (Prev 100):    {m['mean_prev_100']:.4f}\n")
                    f.write(f"   Max Reward:         {m['max_reward']:.4f}\n")
                    f.write(f"   Min Reward:         {m['min_reward']:.4f}\n")
                    f.write(f"   Improvement:        {m['improvement']:.4f}\n")
                    f.write(f"   Improvement %:      {m['improvement_pct']:.2f}%\n")
                    f.write(f"   Is Stable:          {m['is_stable']}\n")
                    f.write(f"   Is Improving:       {m['is_improving']}\n")
                
                f.write("\n")
            
            # توصیه‌ها
            f.write("\n" + "=" * 80 + "\n")
            f.write("💡 RECOMMENDATIONS:\n")
            f.write("=" * 80 + "\n\n")
            
            for level_name, analysis in results_dict.items():
                if analysis['status'] == 'poor_performance':
                    f.write(f"❌ {level_name}: نیاز به آموزش بیشتر (500-1000 اپیزود دیگر)\n")
                elif analysis['status'] == 'stuck':
                    f.write(f"⚠️ {level_name}: بررسی hyperparameters (learning rate, noise)\n")
                elif analysis['status'] == 'improving':
                    f.write(f"📈 {level_name}: ادامه آموزش (200-300 اپیزود دیگر)\n")
                elif analysis['status'] == 'converged':
                    f.write(f"✅ {level_name}: آماده برای ارزیابی نهایی\n")
        
        print(f"\n✅ گزارش ذخیره شد: {report_path}\n")
    
    def run_analysis(self):
        """اجرای تحلیل کامل"""
        
        print("\n" + "=" * 80)
        print("TRAINING STATUS ANALYSIS")
        print("=" * 80 + "\n")
        
        # بارگذاری همه فایل‌ها
        all_data = self.load_all_training_files()
        
        if not all_data:
            print("\n❌ هیچ فایلی برای تحلیل یافت نشد!")
            return {}
        
        print("\n" + "-" * 80 + "\n")
        
        results = {}
        
        for level_name, data in all_data.items():
            print(f"📊 تحلیل {level_name}...")
            
            # استخراج rewards
            rewards = data['results']['training_history']['rewards']
            total_eps = data['results']['total_episodes']
            
            print(f"   📈 تعداد اپیزودها: {total_eps}")
            print(f"   📊 میانگین Reward: {np.mean(rewards):.4f}")
            print(f"   🏆 بهترین Reward: {np.max(rewards):.4f}")
            print(f"   📉 بدترین Reward: {np.min(rewards):.4f}")
            
            # تحلیل همگرایی
            convergence = self.analyze_convergence(rewards)
            results[level_name] = convergence
            
            print(f"   {convergence['message']}")
            
            # رسم نمودار
            self.plot_level_analysis(level_name, data)
            print()
        
        # تولید گزارش
        self.generate_report(results)
        
        print("=" * 80)
        print("✅ تحلیل کامل شد!")
        print("=" * 80 + "\n")
        
        return results


def main():
    """تابع اصلی"""
    
    analyzer = TrainingAnalyzer()
    results = analyzer.run_analysis()
    
    if results:
        print("📋 خلاصه نتایج:\n")
        for level, analysis in results.items():
            print(f"   {level}: {analysis['message']}")


if __name__ == '__main__':
    main()
