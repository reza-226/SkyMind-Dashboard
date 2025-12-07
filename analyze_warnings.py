# فایل: analyze_warnings.py

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class WarningLogAnalyzer:
    """تحلیل WARNING logs از MADDPG Training"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.data = []
        
    def parse_log(self):
        """پارس کردن WARNING logs"""
        
        print(f"\n📖 خواندن فایل: {self.log_file}")
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_episode = None
        current_level = None
        actions_buffer = []
        
        for line in lines:
            # شناسایی Progress Bar
            # Training Level1:  34%|███| 1682/5000 [1:55:18<3:43:14,  4.04s/it]
            progress_match = re.search(
                r'Training Level(\d+):\s+(\d+)%.*?(\d+)/(\d+).*?\[.*?,\s+([\d\.]+)s/it\]',
                line
            )
            
            if progress_match:
                if actions_buffer and current_episode:
                    # ذخیره Episode قبلی
                    self.data.append({
                        'episode': current_episode,
                        'level': current_level,
                        'actions': np.array(actions_buffer),
                        'num_actions': len(actions_buffer)
                    })
                    actions_buffer = []
                
                current_level = int(progress_match.group(1))
                current_episode = int(progress_match.group(3))
                continue
            
            # استخراج Actions
            # [0.975 0.64844537 0.56714371 0.025 0.75559273]
            action_match = re.search(r'action \[([\d\.\s]+)\]', line)
            
            if action_match and current_episode:
                action_str = action_match.group(1)
                actions = [float(x) for x in action_str.split()]
                
                if len(actions) == 5:  # 5 اکشن برای 2 Agent
                    actions_buffer.append(actions)
        
        # ذخیره Episode آخر
        if actions_buffer and current_episode:
            self.data.append({
                'episode': current_episode,
                'level': current_level,
                'actions': np.array(actions_buffer),
                'num_actions': len(actions_buffer)
            })
        
        print(f"✅ استخراج شد: {len(self.data)} Episode")
        return len(self.data) > 0
    
    def analyze(self):
        """تحلیل تنوع اکشن‌ها"""
        
        results = []
        
        for item in self.data:
            actions = item['actions']
            
            # محاسبات آماری
            overall_std = np.std(actions)
            mean_std = np.mean(np.std(actions, axis=0))
            mean_range = np.mean(np.ptp(actions, axis=0))
            
            # تحلیل Agent 0 (فرض: اکشن اول برای Agent 0)
            agent0_actions = actions[:, 0]
            agent0_mean = np.mean(agent0_actions)
            agent0_std = np.std(agent0_actions)
            
            # Agent 1
            agent1_actions = actions[:, 1]
            agent1_mean = np.mean(agent1_actions)
            agent1_std = np.std(agent1_actions)
            
            # تشخیص Freeze
            is_frozen = overall_std < 0.01
            
            results.append({
                'Episode': item['episode'],
                'Level': item['level'],
                'Overall_Std': overall_std,
                'Mean_Std': mean_std,
                'Mean_Range': mean_range,
                'Agent0_Mean': agent0_mean,
                'Agent0_Std': agent0_std,
                'Agent1_Mean': agent1_mean,
                'Agent1_Std': agent1_std,
                'Num_Actions': item['num_actions'],
                'Status': '🚨 FROZEN' if is_frozen else '✅ OK'
            })
        
        return pd.DataFrame(results)
    
    def plot(self, df):
        """رسم نمودارها"""
        
        if df.empty:
            print("⚠️ داده‌ای نیست!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MADDPG: Action Diversity Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Overall Diversity
        ax1 = axes[0, 0]
        ax1.plot(df['Episode'], df['Overall_Std'], 'b-', linewidth=2)
        ax1.axhline(y=0.01, color='r', linestyle='--', linewidth=2, label='Freeze Threshold')
        ax1.fill_between(df['Episode'], 0, 0.01, color='red', alpha=0.1)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Overall Std Dev')
        ax1.set_title('Overall Action Diversity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Agent 0 vs Agent 1
        ax2 = axes[0, 1]
        ax2.plot(df['Episode'], df['Agent0_Mean'], 'purple', label='Agent 0', linewidth=2)
        ax2.plot(df['Episode'], df['Agent1_Mean'], 'orange', label='Agent 1', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Mean Action')
        ax2.set_title('Agent Mean Actions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Agent Std Dev
        ax3 = axes[1, 0]
        ax3.plot(df['Episode'], df['Agent0_Std'], 'purple', label='Agent 0 Std', linewidth=2)
        ax3.plot(df['Episode'], df['Agent1_Std'], 'orange', label='Agent 1 Std', linewidth=2)
        ax3.axhline(y=0.01, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Std Dev')
        ax3.set_title('Agent Action Variability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Status
        ax4 = axes[1, 1]
        frozen_episodes = df[df['Status'] == '🚨 FROZEN']['Episode'].tolist()
        ok_episodes = df[df['Status'] == '✅ OK']['Episode'].tolist()
        
        ax4.scatter(frozen_episodes, [1]*len(frozen_episodes), 
                   c='red', s=100, alpha=0.6, label='Frozen')
        ax4.scatter(ok_episodes, [0]*len(ok_episodes), 
                   c='green', s=100, alpha=0.6, label='OK')
        ax4.set_xlabel('Episode')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['OK', 'FROZEN'])
        ax4.set_title('Episode Status')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('action_diversity_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✅ نمودار ذخیره شد: action_diversity_analysis.png")
    
    def report(self, df):
        """گزارش نهایی"""
        
        print("\n" + "="*70)
        print("📊 خلاصه تحلیل")
        print("="*70)
        
        print(f"\n📈 آمار کلی:")
        print(f"  • تعداد Episodes: {len(df)}")
        print(f"  • Episodes Frozen: {len(df[df['Status'] == '🚨 FROZEN'])}")
        print(f"  • Episodes OK: {len(df[df['Status'] == '✅ OK'])}")
        
        print(f"\n📊 آمار تنوع:")
        print(df[['Overall_Std', 'Mean_Std', 'Mean_Range']].describe())
        
        print(f"\n🤖 آمار Agents:")
        print(df[['Agent0_Mean', 'Agent0_Std', 'Agent1_Mean', 'Agent1_Std']].describe())
        
        # Episodes بحرانی
        frozen = df[df['Status'] == '🚨 FROZEN']
        if not frozen.empty:
            print(f"\n🚨 Episodes بحرانی (Frozen):")
            print(frozen[['Episode', 'Level', 'Overall_Std', 
                         'Agent0_Mean', 'Agent1_Mean']].to_string(index=False))
        
        # ذخیره CSV
        csv_file = 'action_diversity.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n✅ داده‌ها ذخیره شد: {csv_file}")

def main():
    print("🚀 MADDPG WARNING Log Analyzer")
    print("="*70)
    
    # لیست فایل‌های txt
    import os
    txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
    
    if not txt_files:
        print("❌ هیچ فایل txt پیدا نشد!")
        return
    
    print("\n📋 فایل‌های موجود:")
    for i, f in enumerate(txt_files, 1):
        size = os.path.getsize(f) / 1024
        print(f"  {i}. {f} ({size:.1f} KB)")
    
    choice = input("\n❓ شماره فایل لاگ (حاوی WARNING): ")
    
    if not choice.isdigit() or not (1 <= int(choice) <= len(txt_files)):
        print("❌ انتخاب نامعتبر!")
        return
    
    log_file = txt_files[int(choice) - 1]
    
    # تحلیل
    analyzer = WarningLogAnalyzer(log_file)
    
    if not analyzer.parse_log():
        print("❌ پارس نشد! فرمت متفاوته.")
        return
    
    df = analyzer.analyze()
    
    if df.empty:
        print("❌ داده‌ای استخراج نشد!")
        return
    
    # گزارش و نمودار
    analyzer.report(df)
    analyzer.plot(df)
    
    print("\n🎉 تحلیل کامل شد!")

if __name__ == "__main__":
    main()
