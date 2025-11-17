"""
Scenario Comparison and Analysis
================================
مقایسه و تحلیل نتایج سناریوهای مختلف
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


class ScenarioComparator:
    """مقایسه‌گر نتایج سناریوها"""
    
    def __init__(self, results_dir: str = "results/scenarios"):
        self.results_dir = Path(results_dir)
        self.scenarios = []
        self.data = {}
        
        self._load_all_scenarios()
    
    def _load_all_scenarios(self):
        """بارگذاری تمام نتایج"""
        scenario_dirs = [
            'scenario_none',
            'scenario_moderate', 
            'scenario_complex'
        ]
        
        for scenario_name in scenario_dirs:
            scenario_path = self.results_dir / scenario_name
            
            if not scenario_path.exists():
                print(f"⚠️ سناریو {scenario_name} پیدا نشد!")
                continue
            
            # بارگذاری متریک‌ها
            metrics_file = scenario_path / 'training_metrics.npz'
            summary_file = scenario_path / 'summary.json'
            
            if metrics_file.exists():
                metrics = np.load(metrics_file)
                
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                
                self.scenarios.append(scenario_name)
                self.data[scenario_name] = {
                    'metrics': metrics,
                    'summary': summary
                }
                
                print(f"✅ بارگذاری {scenario_name}")
    
    def generate_comparison_plots(self):
        """تولید نمودارهای مقایسه‌ای"""
        
        if len(self.scenarios) < 2:
            print("⚠️ حداقل 2 سناریو نیاز است!")
            return
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # رنگ‌ها برای هر سناریو
        colors = {
            'scenario_none': '#2ecc71',
            'scenario_moderate': '#3498db',
            'scenario_complex': '#e74c3c'
        }
        
        labels = {
            'scenario_none': 'بدون مانع',
            'scenario_moderate': 'موانع متوسط',
            'scenario_complex': 'موانع پیچیده'
        }
        
        # 1. Episode Rewards Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        for scenario in self.scenarios:
            rewards = self.data[scenario]['metrics']['episode_rewards']
            
            # Smoothing
            window = 50
            if len(rewards) > window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(rewards)), smoothed,
                        color=colors[scenario], linewidth=2.5, 
                        label=labels[scenario], alpha=0.9)
        
        ax1.set_xlabel('Episode', fontsize=12, weight='bold')
        ax1.set_ylabel('Average Reward', fontsize=12, weight='bold')
        ax1.set_title('مقایسه Reward در سناریوهای مختلف', 
                     fontsize=14, weight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Final Performance Bar Chart
        ax2 = fig.add_subplot(gs[0, 2])
        final_rewards = [
            self.data[s]['summary']['final_avg_reward'] 
            for s in self.scenarios
        ]
        bars = ax2.bar(range(len(self.scenarios)), final_rewards,
                      color=[colors[s] for s in self.scenarios],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax2.set_xticks(range(len(self.scenarios)))
        ax2.set_xticklabels([labels[s] for s in self.scenarios], 
                           rotation=15, ha='right')
        ax2.set_ylabel('Final Avg Reward (Last 100)', fontsize=10, weight='bold')
        ax2.set_title('مقایسه عملکرد نهایی', fontsize=12, weight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 3. Collisions Comparison
        ax3 = fig.add_subplot(gs[1, 0])
        for scenario in self.scenarios:
            collisions = self.data[scenario]['metrics']['collisions']
            if len(collisions) > 50:
                smoothed = np.convolve(collisions, np.ones(50)/50, mode='valid')
                ax3.plot(range(49, len(collisions)), smoothed,
                        color=colors[scenario], linewidth=2, 
                        label=labels[scenario], alpha=0.9)
        
        ax3.set_xlabel('Episode', fontsize=10)
        ax3.set_ylabel('Collisions', fontsize=10)
        ax3.set_title('تعداد برخوردها', fontsize=12, weight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Tasks Completed
        ax4 = fig.add_subplot(gs[1, 1])
        for scenario in self.scenarios:
            tasks = self.data[scenario]['metrics']['tasks_completed']
            if len(tasks) > 50:
                smoothed = np.convolve(tasks, np.ones(50)/50, mode='valid')
                ax4.plot(range(49, len(tasks)), smoothed,
                        color=colors[scenario], linewidth=2, 
                        label=labels[scenario], alpha=0.9)
        
        ax4.set_xlabel('Episode', fontsize=10)
        ax4.set_ylabel('Tasks Completed', fontsize=10)
        ax4.set_title('وظایف تکمیل شده', fontsize=12, weight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Energy Consumption
        ax5 = fig.add_subplot(gs[1, 2])
        for scenario in self.scenarios:
            energy = self.data[scenario]['metrics']['energy_consumed']
            if len(energy) > 50:
                smoothed = np.convolve(energy, np.ones(50)/50, mode='valid')
                ax5.plot(range(49, len(energy)), smoothed,
                        color=colors[scenario], linewidth=2, 
                        label=labels[scenario], alpha=0.9)
        
        ax5.set_xlabel('Episode', fontsize=10)
        ax5.set_ylabel('Energy (J)', fontsize=10)
        ax5.set_title('مصرف انرژی', fontsize=12, weight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # 6. Collision Risk
        ax6 = fig.add_subplot(gs[2, 0])
        for scenario in self.scenarios:
            risks = self.data[scenario]['metrics']['collision_risks']
            if len(risks) > 50:
                smoothed = np.convolve(risks, np.ones(50)/50, mode='valid')
                ax6.plot(range(49, len(risks)), smoothed,
                        color=colors[scenario], linewidth=2, 
                        label=labels[scenario], alpha=0.9)
        
        ax6.set_xlabel('Episode', fontsize=10)
        ax6.set_ylabel('Risk Level', fontsize=10)
        ax6.set_title('ریسک برخورد', fontsize=12, weight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. Summary Statistics Table
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.axis('tight')
        ax7.axis('off')
        
        table_data = []
        for scenario in self.scenarios:
            summary = self.data[scenario]['summary']
            table_data.append([
                labels[scenario],
                f"{summary['final_avg_reward']:.2f}",
                f"{summary['avg_collisions']:.2f}",
                f"{summary['avg_tasks_completed']:.2f}",
                f"{summary['total_energy_consumed']:.2f}"
            ])
        
        table = ax7.table(
            cellText=table_data,
            colLabels=['سناریو', 'Avg Reward', 'Avg Collisions', 
                      'Avg Tasks', 'Total Energy'],
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.15, 0.15, 0.15, 0.15]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # رنگ‌آمیزی header
        for i in range(5):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # رنگ‌آمیزی ردیف‌ها
        for i, scenario in enumerate(self.scenarios, 1):
            table[(i, 0)].set_facecolor(colors[scenario])
            table[(i, 0)].set_text_props(weight='bold', color='white')
        
        ax7.set_title('خلاصه آماری', fontsize=14, weight='bold', pad=20)
        
        plt.suptitle('مقایسه جامع سناریوهای موانع - پروژه SkyMind',
                    fontsize=16, weight='bold', y=0.995)
        
        # ذخیره
        save_path = self.results_dir / 'scenarios_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 نمودار مقایسه در {save_path} ذخیره شد")
        
        plt.show()
    
    def generate_statistics_table(self):
        """تولید جدول آماری دقیق"""
        
        stats_data = []
        
        for scenario in self.scenarios:
            summary = self.data[scenario]['summary']
            metrics = self.data[scenario]['metrics']
            
            rewards = metrics['episode_rewards']
            collisions = metrics['collisions']
            tasks = metrics['tasks_completed']
            
            stats_data.append({
                'Scenario': scenario.replace('scenario_', '').title(),
                'Final Reward': summary['final_avg_reward'],
                'Best Reward': summary['best_reward'],
                'Avg Collisions': summary['avg_collisions'],
                'Total Tasks': summary['avg_tasks_completed'],
                'Total Energy': summary['total_energy_consumed'],
                'Std Reward': float(np.std(rewards)),
                'Success Rate': float(np.mean(np.array(tasks) > 5))
            })
        
        df = pd.DataFrame(stats_data)
        
        # ذخیره CSV
        csv_path = self.results_dir / 'comparison_statistics.csv'
        df.to_csv(csv_path, index=False, float_format='%.3f')
        print(f"📄 جدول آماری در {csv_path} ذخیره شد")
        
        # چاپ جدول
        print("\n" + "="*80)
        print("📊 جدول آماری مقایسه سناریوها")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        return df
    
    def print_summary(self):
        """چاپ خلاصه نتایج"""
        print("\n" + "="*80)
        print("🎯 خلاصه نتایج مقایسه")
        print("="*80 + "\n")
        
        best_scenario = None
        best_reward = -np.inf
        
        for scenario in self.scenarios:
            summary