"""
Output Manager with Enhanced Metrics Support
مدیریت خروجی‌ها با پشتیبانی از متریک‌های پیشرفته
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class OutputManager:
    """
    مدیریت ساختاریافته خروجی‌های آموزش با متریک‌های تخصصی
    """
    
    def __init__(self, base_dir: str = "results", level: int = 1, 
                 difficulty: str = "easy", run_id: Optional[str] = None):
        """
        Args:
            base_dir: دایرکتوری پایه برای ذخیره نتایج
            level: شماره سطح (1-5)
            difficulty: سختی سطح (easy/medium/hard)
            run_id: شناسه یکتا برای اجرا (اگر None باشد، timestamp استفاده می‌شود)
        """
        self.base_dir = Path(base_dir)
        self.level = level
        self.difficulty = difficulty
        
        # ایجاد run_id یکتا
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"run_{timestamp}"
        else:
            self.run_id = run_id
        
        # ساخت ساختار دایرکتوری
        self.run_dir = self.base_dir / f"level{level}_{difficulty}" / self.run_id
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.best_models_dir = self.run_dir / "best_models"
        self.logs_dir = self.run_dir / "logs"
        self.plots_dir = self.run_dir / "plots"
        self.configs_dir = self.run_dir / "configs"
        self.detailed_metrics_dir = self.run_dir / "detailed_metrics"
        
        self._create_directories()
        
        # لیست برای ذخیره تاریخچه
        self.training_history = []
        self.detailed_history = []
        
    def _create_directories(self):
        """ایجاد تمام دایرکتوری‌های مورد نیاز"""
        for directory in [self.run_dir, self.checkpoints_dir, self.best_models_dir,
                         self.logs_dir, self.plots_dir, self.configs_dir,
                         self.detailed_metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Output structure created at: {self.run_dir}")
    
    def save_config(self, config: Dict[str, Any]):
        """ذخیره تنظیمات آموزش"""
        config_file = self.configs_dir / "training_config.json"
        
        # تبدیل config به فرمت قابل ذخیره
        serializable_config = self._make_serializable(config)
        
        with open(config_file, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        
        print(f"✅ Config saved to: {config_file}")
    
    def _make_serializable(self, obj):
        """تبدیل اشیاء به فرمت قابل سریالایز"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def save_checkpoint(self, checkpoint: Dict[str, Any], episode: int):
        """
        ذخیره checkpoint
        
        Args:
            checkpoint: دیکشنری شامل state های مدل و optimizer
            episode: شماره اپیزود
        """
        checkpoint_file = self.checkpoints_dir / f"checkpoint_episode_{episode}.pt"
        
        import torch
        torch.save(checkpoint, checkpoint_file)
        
        print(f"💾 Checkpoint saved: episode {episode}")
    
    def save_best_model(self, checkpoint: Dict[str, Any], episode: int, metric_value: float):
        """
        ذخیره بهترین مدل
        
        Args:
            checkpoint: دیکشنری شامل state های مدل
            episode: شماره اپیزود
            metric_value: مقدار متریک (مثلاً reward)
        """
        best_model_file = self.best_models_dir / f"best_model_episode_{episode}.pt"
        
        checkpoint['best_metric_value'] = metric_value
        checkpoint['best_episode'] = episode
        
        import torch
        torch.save(checkpoint, best_model_file)
        
        print(f"🏆 Best model saved: episode {episode}, metric = {metric_value:.4f}")
    
    def save_training_history(self, metrics: Dict[str, Any]):
        """
        ذخیره متریک‌های یک اپیزود
        
        Args:
            metrics: دیکشنری شامل تمام متریک‌های اپیزود
        """
        self.training_history.append(metrics)
        
        # ذخیره به JSON
        history_json = self.logs_dir / "training_history.json"
        with open(history_json, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # ذخیره به CSV
        history_csv = self.logs_dir / "training_history.csv"
        df = pd.DataFrame(self.training_history)
        df.to_csv(history_csv, index=False)
    
    def save_detailed_metrics(self, episode: int, metrics: Dict[str, Any]):
        """
        ذخیره متریک‌های تفصیلی یک اپیزود
        
        Args:
            episode: شماره اپیزود
            metrics: دیکشنری شامل متریک‌های تفصیلی
        """
        self.detailed_history.append(metrics)
        
        # ذخیره به فایل جداگانه برای هر اپیزود
        episode_file = self.detailed_metrics_dir / f"episode_{episode}_detailed.json"
        with open(episode_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # ذخیره تجمعی
        detailed_csv = self.logs_dir / "detailed_metrics.csv"
        df = pd.DataFrame(self.detailed_history)
        df.to_csv(detailed_csv, index=False)
    
    def generate_training_plots(self):
        """تولید نمودارهای آموزش"""
        if not self.training_history:
            print("⚠️ No training history to plot")
            return
        
        df = pd.DataFrame(self.training_history)
        
        # 1. Basic Learning Curves
        self._plot_learning_curves(df)
        
        # 2. Specialized Metrics
        self._plot_specialized_metrics(df)
        
        # 3. Correlation Analysis
        self._plot_correlation_analysis(df)
        
        print(f"📊 Plots saved to: {self.plots_dir}")
    
    def _plot_learning_curves(self, df: pd.DataFrame):
        """رسم منحنی‌های یادگیری پایه"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward
        axes[0, 0].plot(df['episode'], df['avg_reward'], alpha=0.5, label='Raw')
        if len(df) > 50:
            axes[0, 0].plot(df['episode'], df['avg_reward'].rolling(50).mean(), 
                           'r-', linewidth=2, label='MA-50')
        axes[0, 0].set_title('📈 Average Reward', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Actor Loss
        if 'actor_loss' in df.columns:
            axes[0, 1].plot(df['episode'], df['actor_loss'], alpha=0.5)
            if len(df) > 50:
                axes[0, 1].plot(df['episode'], df['actor_loss'].rolling(50).mean(),
                               'g-', linewidth=2, label='MA-50')
            axes[0, 1].set_title('🎯 Actor Loss', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Critic Loss
        if 'critic_loss' in df.columns:
            axes[1, 0].plot(df['episode'], df['critic_loss'], alpha=0.5)
            if len(df) > 50:
                axes[1, 0].plot(df['episode'], df['critic_loss'].rolling(50).mean(),
                               'b-', linewidth=2, label='MA-50')
            axes[1, 0].set_title('🎯 Critic Loss', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Epsilon
        if 'epsilon' in df.columns:
            axes[1, 1].plot(df['episode'], df['epsilon'], 'purple', linewidth=2)
            axes[1, 1].set_title('🔍 Exploration (Epsilon)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_specialized_metrics(self, df: pd.DataFrame):
        """رسم متریک‌های تخصصی"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 13))
        
        # 1. Energy Consumption
        if 'total_energy' in df.columns:
            axes[0, 0].plot(df['episode'], df['total_energy'], alpha=0.6, color='orange')
            if len(df) > 50:
                axes[0, 0].plot(df['episode'], df['total_energy'].rolling(50).mean(),
                               'r-', linewidth=2, label='MA-50')
            axes[0, 0].set_title('🔋 Total Energy Consumption', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Energy (Joules)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Average Latency
        if 'avg_latency' in df.columns:
            axes[0, 1].plot(df['episode'], df['avg_latency'], alpha=0.6, color='green')
            if len(df) > 50:
                axes[0, 1].plot(df['episode'], df['avg_latency'].rolling(50).mean(),
                               'darkgreen', linewidth=2, label='MA-50')
            axes[0, 1].set_title('⏱️ Average Latency', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Latency (ms)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Task Success Rate
        if 'task_success_rate' in df.columns:
            axes[1, 0].plot(df['episode'], df['task_success_rate'] * 100, alpha=0.6, color='blue')
            if len(df) > 50:
                axes[1, 0].plot(df['episode'], (df['task_success_rate'] * 100).rolling(50).mean(),
                               'darkblue', linewidth=2, label='MA-50')
            axes[1, 0].set_title('✅ Task Success Rate', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Success Rate (%)')
            axes[1, 0].set_ylim([0, 105])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Collision Rate
        if 'collisions' in df.columns:
            axes[1, 1].plot(df['episode'], df['collisions'], 'r.', alpha=0.4, markersize=3)
            if len(df) > 50:
                axes[1, 1].plot(df['episode'], df['collisions'].rolling(50).mean(),
                               'darkred', linewidth=2, label='MA-50')
            axes[1, 1].set_title('💥 Collision Rate', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Collisions per Episode')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Energy Efficiency
        if 'energy_efficiency' in df.columns:
            axes[2, 0].plot(df['episode'], df['energy_efficiency'], alpha=0.6, color='purple')
            if len(df) > 50:
                axes[2, 0].plot(df['episode'], df['energy_efficiency'].rolling(50).mean(),
                               'indigo', linewidth=2, label='MA-50')
            axes[2, 0].set_title('⚡ Energy Efficiency (Tasks/J)', fontsize=12, fontweight='bold')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Efficiency')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Offloading Ratio
        if 'offloading_ratio' in df.columns:
            axes[2, 1].plot(df['episode'], df['offloading_ratio'] * 100, alpha=0.6, color='teal')
            if len(df) > 50:
                axes[2, 1].plot(df['episode'], (df['offloading_ratio'] * 100).rolling(50).mean(),
                               'darkcyan', linewidth=2, label='MA-50')
            axes[2, 1].set_title('📡 Offloading Ratio', fontsize=12, fontweight='bold')
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Offloading Ratio (%)')
            axes[2, 1].set_ylim([0, 105])
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'specialized_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, df: pd.DataFrame):
        """تحلیل همبستگی متریک‌ها"""
        # انتخاب ستون‌های عددی
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # حذف ستون‌های غیرضروری
        exclude_cols = ['episode', 'level', 'timestamp']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) < 2:
            return
        
        # محاسبه ماتریس همبستگی
        corr_matrix = df[numeric_cols].corr()
        
        # رسم heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('📊 Correlation Matrix of Training Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """تولید گزارش تحلیلی جامع"""
        if not self.training_history:
            return {}
        
        df = pd.DataFrame(self.training_history)
        
        report = {
            'training_summary': self._analyze_training_progress(df),
            'uav_analysis': self._analyze_uav_metrics(df),
            'task_analysis': self._analyze_task_metrics(df),
            'energy_analysis': self._analyze_energy_metrics(df),
            'safety_analysis': self._analyze_safety_metrics(df),
            'performance_milestones': self._identify_milestones(df),
        }
        
        # ذخیره گزارش
        report_file = self.run_dir / 'analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # تولید گزارش متنی
        self._generate_text_report(report)
        
        return report
    
    def _analyze_training_progress(self, df: pd.DataFrame) -> Dict:
        """تحلیل پیشرفت آموزش"""
        return {
            'total_episodes': len(df),
            'best_episode': int(df.loc[df['avg_reward'].idxmax(), 'episode']),
            'best_reward': float(df['avg_reward'].max()),
            'worst_reward': float(df['avg_reward'].min()),
            'final_reward': float(df['avg_reward'].iloc[-1]),
            'avg_reward': float(df['avg_reward'].mean()),
            'reward_std': float(df['avg_reward'].std()),
            'improvement': float(df['avg_reward'].iloc[-100:].mean() - df['avg_reward'].iloc[:100].mean()) 
                          if len(df) > 200 else 0.0,
        }
    
    def _analyze_uav_metrics(self, df: pd.DataFrame) -> Dict:
        """تحلیل متریک‌های UAV"""
        analysis = {}
        
        if 'num_uavs' in df.columns:
            analysis['avg_active_uavs'] = float(df['num_uavs'].mean())
        
        if 'avg_uav_battery' in df.columns:
            analysis['avg_battery_level'] = float(df['avg_uav_battery'].mean())
            analysis['min_battery_level'] = float(df['avg_uav_battery'].min())
            analysis['battery_depletion_rate'] = self._calculate_depletion_rate(df, 'avg_uav_battery')
        
        return analysis
    
    def _analyze_task_metrics(self, df: pd.DataFrame) -> Dict:
        """تحلیل متریک‌های وظایف"""
        analysis = {}
        
        if 'tasks_completed' in df.columns:
            analysis['total_tasks_completed'] = int(df['tasks_completed'].sum())
            analysis['avg_tasks_per_episode'] = float(df['tasks_completed'].mean())
        
        if 'task_success_rate' in df.columns:
            analysis['avg_success_rate'] = float(df['task_success_rate'].mean() * 100)
            analysis['success_rate_improvement'] = float(
                (df['task_success_rate'].iloc[-100:].mean() - 
                 df['task_success_rate'].iloc[:100].mean()) * 100
            ) if len(df) > 200 else 0.0
        
        if 'avg_latency' in df.columns:
            analysis['avg_latency'] = float(df['avg_latency'].mean())
            analysis['latency_reduction'] = self._calculate_latency_reduction(df)
        
        if 'offloading_ratio' in df.columns:
            analysis['avg_offloading_ratio'] = float(df['offloading_ratio'].mean() * 100)
        
        return analysis
    
    def _analyze_energy_metrics(self, df: pd.DataFrame) -> Dict:
        """تحلیل متریک‌های انرژی"""
        analysis = {}
        
        if 'total_energy' in df.columns:
            analysis['total_energy_consumed'] = float(df['total_energy'].sum())
            analysis['avg_energy_per_episode'] = float(df['total_energy'].mean())
            analysis['energy_efficiency_improvement'] = float(
                (df['total_energy'].iloc[:100].mean() - df['total_energy'].iloc[-100:].mean()) /
                df['total_energy'].iloc[:100].mean() * 100
            ) if len(df) > 200 and df['total_energy'].iloc[:100].mean() > 0 else 0.0
        
        if 'energy_efficiency' in df.columns:
            analysis['avg_energy_efficiency'] = float(df['energy_efficiency'].mean())
            analysis['max_energy_efficiency'] = float(df['energy_efficiency'].max())
        
        return analysis
    
    def _analyze_safety_metrics(self, df: pd.DataFrame) -> Dict:
        """تحلیل متریک‌های ایمنی"""
        analysis = {}
        
        if 'collisions' in df.columns:
            analysis['total_collisions'] = int(df['collisions'].sum())
            analysis['collision_rate'] = float(df['collisions'].mean())
            analysis['collision_free_episodes'] = int((df['collisions'] == 0).sum())
            analysis['collision_free_rate'] = float((df['collisions'] == 0).mean() * 100)
        
        if 'min_obstacle_distance' in df.columns:
            valid_distances = df[df['min_obstacle_distance'] < float('inf')]['min_obstacle_distance']
            if len(valid_distances) > 0:
                analysis['avg_obstacle_clearance'] = float(valid_distances.mean())
                analysis['min_obstacle_clearance'] = float(valid_distances.min())
        
        return analysis
    
    def _calculate_depletion_rate(self, df: pd.DataFrame, column: str) -> float:
        """محاسبه نرخ کاهش (مثل باتری)"""
        if len(df) < 100:
            return 0.0
        
        initial = df[column].iloc[:50].mean()
        final = df[column].iloc[-50:].mean()
        
        if initial > 0:
            return float((initial - final) / initial * 100)
        return 0.0
    
    def _calculate_latency_reduction(self, df: pd.DataFrame) -> float:
        """محاسبه کاهش تأخیر"""
        if 'avg_latency' not in df.columns or len(df) < 200:
            return 0.0
        
        baseline = df['avg_latency'].iloc[:100].mean()
        current = df['avg_latency'].iloc[-100:].mean()
        
        if baseline > 0:
            return float((baseline - current) / baseline * 100)
        return 0.0
    
    def _identify_milestones(self, df: pd.DataFrame) -> Dict:
        """شناسایی نقاط عطف مهم"""
        milestones = {}
        
        # بهترین عملکرد
        if 'avg_reward' in df.columns:
            best_idx = df['avg_reward'].idxmax()
            milestones['best_performance'] = {
                'episode': int(df.loc[best_idx, 'episode']),
                'reward': float(df.loc[best_idx, 'avg_reward']),
            }
        
        # اولین موفقیت (reward > threshold)
        reward_threshold = -10.0  # قابل تنظیم
        if 'avg_reward' in df.columns:
            success_episodes = df[df['avg_reward'] > reward_threshold]
            if len(success_episodes) > 0:
                first_success = success_episodes.iloc[0]
                milestones['first_success'] = {
                    'episode': int(first_success['episode']),
                    'reward': float(first_success['avg_reward']),
                }
        
        # کمترین برخورد
        if 'collisions' in df.columns:
            collision_free = df[df['collisions'] == 0]
            if len(collision_free) > 0:
                milestones['first_collision_free'] = {
                    'episode': int(collision_free.iloc[0]['episode']),
                }
                milestones['collision_free_count'] = len(collision_free)
        
        return milestones
    
    def _generate_text_report(self, report: Dict):
        """تولید گزارش متنی قابل خواندن"""
        report_file = self.run_dir / 'REPORT.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("📊 TRAINING ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Level: {self.level} ({self.difficulty})\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training Summary
            f.write("-" * 80 + "\n")
            f.write("🎯 TRAINING SUMMARY\n")
            f.write("-" * 80 + "\n")
            ts = report.get('training_summary', {})
            f.write(f"Total Episodes: {ts.get('total_episodes', 0)}\n")
            f.write(f"Best Episode: {ts.get('best_episode', 0)}\n")
            f.write(f"Best Reward: {ts.get('best_reward', 0):.4f}\n")
            f.write(f"Average Reward: {ts.get('avg_reward', 0):.4f}\n")
            f.write(f"Improvement: {ts.get('improvement', 0):.4f}\n\n")
            
            # UAV Analysis
            f.write("-" * 80 + "\n")
            f.write("🚁 UAV ANALYSIS\n")
            f.write("-" * 80 + "\n")
            ua = report.get('uav_analysis', {})
            for key, value in ua.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            
            # Task Analysis
            f.write("-" * 80 + "\n")
            f.write("📋 TASK ANALYSIS\n")
            f.write("-" * 80 + "\n")
            ta = report.get('task_analysis', {})
            for key, value in ta.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            
            # Energy Analysis
            f.write("-" * 80 + "\n")
            f.write("⚡ ENERGY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            ea = report.get('energy_analysis', {})
            for key, value in ea.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            
            # Safety Analysis
            f.write("-" * 80 + "\n")
            f.write("🛡️ SAFETY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            sa = report.get('safety_analysis', {})
            for key, value in sa.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            
            # Milestones
            f.write("-" * 80 + "\n")
            f.write("🏆 PERFORMANCE MILESTONES\n")
            f.write("-" * 80 + "\n")
            pm = report.get('performance_milestones', {})
            for milestone, data in pm.items():
                f.write(f"\n{milestone}:\n")
                for key, value in data.items():
                    f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"📄 Text report saved to: {report_file}")
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """پیدا کردن آخرین checkpoint"""
        checkpoints = list(self.checkpoints_dir.glob("checkpoint_episode_*.pt"))
        if not checkpoints:
            return None
        
        # مرتب‌سازی بر اساس شماره اپیزود
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoints[-1]
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None):
        """بارگذاری checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
        
        if checkpoint_path is None:
            print("❌ No checkpoint found")
            return None
        
        import torch
        checkpoint = torch.load(checkpoint_path)
        print(f"✅ Checkpoint loaded from: {checkpoint_path}")
        return checkpoint
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """حذف checkpoint های قدیمی برای ذخیره فضا"""
        checkpoints = list(self.checkpoints_dir.glob("checkpoint_episode_*.pt"))
        
        if len(checkpoints) <= keep_last_n:
            return
        
        # مرتب‌سازی
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # حذف قدیمی‌ها
        for checkpoint in checkpoints[:-keep_last_n]:
            checkpoint.unlink()
            print(f"🗑️ Deleted old checkpoint: {checkpoint.name}")


# ========================================
# Helper Functions
# ========================================

def create_output_structure(base_dir: str = "results", levels: range = range(1, 6)):
    """ایجاد ساختار پایه برای تمام سطوح"""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    difficulties = ['easy', 'medium', 'hard']
    
    for level in levels:
        for difficulty in difficulties:
            level_dir = base_path / f"level{level}_{difficulty}"
            level_dir.mkdir(exist_ok=True)
    
    print(f"✅ Base output structure created at: {base_dir}")


if __name__ == "__main__":
    # تست
    manager = OutputManager(level=1, difficulty="easy")
    
    # تست ذخیره config
    test_config = {
        'network': {'actor_hidden': [128, 64]},
        'training': {'learning_rate': 0.001}
    }
    manager.save_config(test_config)
    
    # تست ذخیره متریک‌ها
    test_metrics = {
        'episode': 1,
        'avg_reward': -15.5,
        'num_uavs': 2,
        'total_energy': 150.0,
        'avg_latency': 25.5,
        'collisions': 0,
    }
    manager.save_training_history(test_metrics)
    
    print("✅ OutputManager test completed!")
