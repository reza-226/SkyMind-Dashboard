# فایل: analyze_training_history.py

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

class TrainingHistoryAnalyzer:
    """تحلیل‌گر جامع Training History"""
    
    def __init__(self, history_path="models/maddpg/training_history.json"):
        self.history_path = Path(history_path)
        self.data = None
        self.df = None
        
    def load_data(self):
        """بارگذاری Training History"""
        print(f"📂 بارگذاری: {self.history_path}")
        
        with open(self.history_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✅ تعداد Episodes: {len(self.data)}")
        
        # نمونه داده
        if self.data:
            first_key = list(self.data.keys())[0]
            print(f"\n📋 نمونه داده (Episode {first_key}):")
            print(json.dumps(self.data[first_key], indent=2, ensure_ascii=False))
    
    def create_dataframe(self):
        """تبدیل به DataFrame"""
        records = []
        
        for episode_key, metrics in self.data.items():
            record = {
                'episode': int(episode_key),
                'stage': metrics.get('stage', 'unknown'),
                'avg_reward': metrics.get('avg_reward', 0),
                'agent_0_reward': metrics.get('rewards', {}).get('agent_0', 0),
                'agent_1_reward': metrics.get('rewards', {}).get('agent_1', 0),
                'actor_loss': metrics.get('actor_loss', 0),
                'critic_loss': metrics.get('critic_loss', 0)
            }
            records.append(record)
        
        self.df = pd.DataFrame(records).sort_values('episode')
        print(f"\n✅ DataFrame ایجاد شد: {self.df.shape}")
        print(f"\n📊 ستون‌های موجود:")
        print(self.df.columns.tolist())
        print(f"\n📈 آمار اولیه:")
        print(self.df.describe())
        
        return self.df
    
    def analyze_stages(self):
        """تحلیل Stages مختلف"""
        print("\n" + "="*80)
        print("🎯 تحلیل Curriculum Stages")
        print("="*80)
        
        stage_stats = self.df.groupby('stage').agg({
            'episode': ['count', 'min', 'max'],
            'avg_reward': ['mean', 'std', 'min', 'max'],
            'actor_loss': 'mean',
            'critic_loss': 'mean'
        }).round(4)
        
        print("\n📊 آمار هر Stage:")
        print(stage_stats)
        
        # یافتن نقطه تغییر Stage
        stage_changes = self.df[self.df['stage'] != self.df['stage'].shift()].copy()
        if len(stage_changes) > 1:
            print(f"\n🔄 تغییرات Stage:")
            for idx, row in stage_changes.iterrows():
                print(f"  Episode {int(row['episode'])}: {row['stage']}")
    
    def analyze_crisis(self):
        """تحلیل بحران Over-Specialization"""
        print("\n" + "="*80)
        print("🔍 تحلیل بحران Over-Specialization")
        print("="*80)
        
        # محاسبه Rolling Statistics
        window = 50
        self.df['reward_ma'] = self.df['avg_reward'].rolling(window).mean()
        self.df['reward_std'] = self.df['avg_reward'].rolling(window).std()
        
        # یافتن Episodes با Reward پایین
        if len(self.df) > window:
            last_100 = self.df.tail(100)
            recent_avg = last_100['avg_reward'].mean()
            overall_avg = self.df['avg_reward'].mean()
            
            print(f"\n📉 مقایسه Reward:")
            print(f"  • میانگین کلی: {overall_avg:.2f}")
            print(f"  • میانگین 100 اپیزود اخیر: {recent_avg:.2f}")
            print(f"  • تفاوت: {recent_avg - overall_avg:.2f} ({(recent_avg/overall_avg - 1)*100:.1f}%)")
            
            if recent_avg < overall_avg * 0.8:
                print("\n🚨 هشدار: کاهش شدید عملکرد در Episodes اخیر!")
            elif recent_avg > overall_avg * 1.2:
                print("\n✅ عملکرد در حال بهبود است!")
        
        # یافتن Episodes با Loss بالا
        if self.df['actor_loss'].max() > 0:
            high_loss = self.df[self.df['actor_loss'] > self.df['actor_loss'].quantile(0.95)]
            if not high_loss.empty:
                print(f"\n⚠️ {len(high_loss)} Episode با Actor Loss بالا (>95th percentile):")
                print(high_loss[['episode', 'stage', 'avg_reward', 'actor_loss']].head(10))
    
    def plot_comprehensive_analysis(self):
        """نمودارهای جامع"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # تنظیمات فونت فارسی (اختیاری)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # 1. Rewards Over Time
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.df['episode'], self.df['avg_reward'], 
                'o', alpha=0.3, markersize=2, label='Episode Reward', color='blue')
        ax1.plot(self.df['episode'], self.df['reward_ma'], 
                'r-', linewidth=2, label='Moving Avg (50)')
        ax1.fill_between(self.df['episode'], 
                         self.df['reward_ma'] - self.df['reward_std'],
                         self.df['reward_ma'] + self.df['reward_std'],
                         alpha=0.2, color='red', label='Std Dev')
        
        # مشخص کردن Stage Changes
        stage_changes = self.df[self.df['stage'] != self.df['stage'].shift()]
        for _, row in stage_changes.iterrows():
            ax1.axvline(x=row['episode'], color='green', 
                       linestyle='--', alpha=0.5, linewidth=1.5)
            ax1.text(row['episode'], ax1.get_ylim()[1], f"{row['stage']}", 
                    rotation=90, va='top', fontsize=9, color='green')
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Average Reward', fontsize=12)
        ax1.set_title('Training Rewards Over Time (with Stage Transitions)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Reward by Stage (Box Plot)
        ax2 = fig.add_subplot(gs[0, 2])
        stages = sorted(self.df['stage'].unique())
        stage_data = [self.df[self.df['stage'] == stage]['avg_reward'] for stage in stages]
        bp = ax2.boxplot(stage_data, labels=stages, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_xlabel('Stage', fontsize=12)
        ax2.set_ylabel('Reward', fontsize=12)
        ax2.set_title('Reward Distribution by Stage', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Actor Loss
        ax3 = fig.add_subplot(gs[1, :2])
        valid_loss = self.df[self.df['actor_loss'] > 0]
        if not valid_loss.empty:
            ax3.plot(valid_loss['episode'], valid_loss['actor_loss'], 
                    'o-', alpha=0.6, markersize=2, color='orange')
            ax3.set_yscale('log')
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('Actor Loss (log scale)', fontsize=12)
        ax3.set_title('Actor Loss Over Time', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Critic Loss
        ax4 = fig.add_subplot(gs[1, 2])
        valid_critic = self.df[self.df['critic_loss'] > 0]
        if not valid_critic.empty:
            ax4.plot(valid_critic['episode'], valid_critic['critic_loss'], 
                    'o-', alpha=0.6, markersize=2, color='purple')
            ax4.set_yscale('log')
        ax4.set_xlabel('Episode', fontsize=12)
        ax4.set_ylabel('Critic Loss (log scale)', fontsize=12)
        ax4.set_title('Critic Loss Over Time', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Agent Rewards Comparison
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(self.df['episode'], self.df['agent_0_reward'], 
                alpha=0.5, label='Agent 0', color='blue')
        ax5.plot(self.df['episode'], self.df['agent_1_reward'], 
                alpha=0.5, label='Agent 1', color='green')
        ax5.plot(self.df['episode'], 
                self.df['agent_0_reward'].rolling(50).mean(), 
                linewidth=2, color='blue')
        ax5.plot(self.df['episode'], 
                self.df['agent_1_reward'].rolling(50).mean(), 
                linewidth=2, color='green')
        ax5.set_xlabel('Episode', fontsize=12)
        ax5.set_ylabel('Reward', fontsize=12)
        ax5.set_title('Individual Agent Rewards', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Reward Improvement Rate
        ax6 = fig.add_subplot(gs[2, 1])
        window = 100
        if len(self.df) > window:
            improvement = self.df['avg_reward'].rolling(window).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / window if len(x) == window else 0
            )
            ax6.plot(self.df['episode'], improvement, 'g-', linewidth=2)
            ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Episode', fontsize=12)
        ax6.set_ylabel('Improvement Rate', fontsize=12)
        ax6.set_title(f'Reward Improvement Rate ({window}-ep window)', 
                     fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 7. Recent Performance (Last 200 episodes)
        ax7 = fig.add_subplot(gs[2, 2])
        last_n = min(200, len(self.df))
        recent = self.df.tail(last_n)
        ax7.hist(recent['avg_reward'], bins=30, 
                color='teal', alpha=0.7, edgecolor='black')
        ax7.axvline(x=recent['avg_reward'].mean(), 
                   color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {recent['avg_reward'].mean():.2f}")
        ax7.set_xlabel('Reward', fontsize=12)
        ax7.set_ylabel('Frequency', fontsize=12)
        ax7.set_title(f'Recent Performance (Last {last_n} Episodes)', 
                     fontsize=14, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('MADDPG Training History - Comprehensive Analysis', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        output_path = 'training_history_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ نمودار ذخیره شد: {output_path}")
        plt.close()
    
    def generate_report(self):
        """تولید گزارش نهایی"""
        
        print("\n" + "="*80)
        print("📊 گزارش نهایی Training History")
        print("="*80)
        
        print(f"\n📈 آمار کلی:")
        print(f"  • تعداد Episodes: {len(self.df)}")
        print(f"  • Episode اول: {self.df['episode'].min()}")
        print(f"  • Episode آخر: {self.df['episode'].max()}")
        
        print(f"\n🎯 Rewards:")
        print(f"  • میانگین: {self.df['avg_reward'].mean():.2f}")
        print(f"  • بهترین: {self.df['avg_reward'].max():.2f}")
        print(f"  • بدترین: {self.df['avg_reward'].min():.2f}")
        print(f"  • انحراف معیار: {self.df['avg_reward'].std():.2f}")
        
        # پیدا کردن بهترین Episode
        best_ep = self.df.loc[self.df['avg_reward'].idxmax()]
        print(f"\n🏆 بهترین Episode: {int(best_ep['episode'])} (Stage: {best_ep['stage']})")
        print(f"  • Reward: {best_ep['avg_reward']:.2f}")
        print(f"  • Agent 0: {best_ep['agent_0_reward']:.2f}")
        print(f"  • Agent 1: {best_ep['agent_1_reward']:.2f}")
        
        # آمار Losses
        if self.df['actor_loss'].max() > 0:
            print(f"\n📉 Actor Loss:")
            valid_actor = self.df[self.df['actor_loss'] > 0]
            print(f"  • میانگین: {valid_actor['actor_loss'].mean():.6f}")
            print(f"  • بیشترین: {valid_actor['actor_loss'].max():.6f}")
            print(f"  • کمترین: {valid_actor['actor_loss'].min():.6f}")
        
        if self.df['critic_loss'].max() > 0:
            print(f"\n📉 Critic Loss:")
            valid_critic = self.df[self.df['critic_loss'] > 0]
            print(f"  • میانگین: {valid_critic['critic_loss'].mean():.6f}")
            print(f"  • بیشترین: {valid_critic['critic_loss'].max():.6f}")
            print(f"  • کمترین: {valid_critic['critic_loss'].min():.6f}")
        
        # تحلیل 100 اپیزود اخیر
        last_100 = self.df.tail(100)
        print(f"\n🔥 عملکرد 100 اپیزود اخیر:")
        print(f"  • میانگین Reward: {last_100['avg_reward'].mean():.2f}")
        print(f"  • بهترین: {last_100['avg_reward'].max():.2f}")
        print(f"  • بدترین: {last_100['avg_reward'].min():.2f}")
        
        # ذخیره CSV
        csv_path = 'training_history_analysis.csv'
        self.df.to_csv(csv_path, index=False)
        print(f"\n✅ CSV ذخیره شد: {csv_path}")
        
        # Summary JSON
        summary = {
            'total_episodes': int(len(self.df)),
            'stages': self.df['stage'].unique().tolist(),
            'best_episode': int(best_ep['episode']),
            'best_reward': float(best_ep['avg_reward']),
            'overall_avg_reward': float(self.df['avg_reward'].mean()),
            'recent_100_avg_reward': float(last_100['avg_reward'].mean())
        }
        
        with open('training_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Summary JSON ذخیره شد: training_summary.json")

def main():
    print("🚀 Training History Analyzer v2.0")
    print("="*80)
    
    analyzer = TrainingHistoryAnalyzer()
    
    try:
        analyzer.load_data()
        analyzer.create_dataframe()
        analyzer.analyze_stages()
        analyzer.analyze_crisis()
        analyzer.plot_comprehensive_analysis()
        analyzer.generate_report()
        
        print("\n" + "="*80)
        print("🎉 تحلیل کامل شد!")
        print("="*80)
        print("\n📁 فایل‌های خروجی:")
        print("  • training_history_analysis.png")
        print("  • training_history_analysis.csv")
        print("  • training_summary.json")
        
    except FileNotFoundError:
        print(f"❌ فایل پیدا نشد: {analyzer.history_path}")
        print("💡 مسیر فعلی را بررسی کنید یا مسیر کامل را وارد کنید.")
    except Exception as e:
        print(f"❌ خطا: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
