"""
سیستم لاگ پیشرفته برای آموزش و داشبورد
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np


class TrainingLogger:
    """
    Logger برای ذخیره تمام اطلاعات آموزش
    """
    def __init__(self, log_dir: str = "logs"):
        """
        Args:
            log_dir: مسیر ذخیره log‌ها
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # ساخت timestamp برای این session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)
        
        # فایل‌های CSV برای ذخیره metrics
        self.episode_log_path = self.session_dir / "episode_metrics.csv"
        self.step_log_path = self.session_dir / "step_metrics.csv"
        self.gnn_log_path = self.session_dir / "gnn_metrics.csv"
        
        # ایجاد فایل‌های CSV با header
        self._init_csv_files()
        
        # Buffer برای ذخیره realtime
        self.episode_buffer = []
        self.step_buffer = []
        self.gnn_buffer = []
        
        print(f"📊 Logger initialized: {self.session_dir}")
    
    def _init_csv_files(self):
        """ایجاد فایل‌های CSV با header"""
        # Episode metrics
        with open(self.episode_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'timestamp', 'total_reward', 'avg_reward',
                'actor_loss', 'critic_loss', 'entropy',
                'episode_length', 'success_rate', 'avg_delay',
                'avg_energy', 'num_offloaded_tasks'
            ])
        
        # Step metrics
        with open(self.step_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'step', 'timestamp', 'agent_id',
                'action', 'reward', 'value', 'entropy',
                'task_id', 'offload_target', 'delay', 'energy'
            ])
        
        # GNN metrics
        with open(self.gnn_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'step', 'timestamp',
                'num_nodes', 'num_edges', 'avg_critical_score',
                'max_critical_score', 'critical_path_length',
                'graph_density'
            ])
    
    def log_step(
        self,
        episode: int,
        step: int,
        agent_id: int,
        action: int,
        reward: float,
        info: Dict
    ):
        """
        لاگ یک step
        
        Args:
            episode: شماره episode
            step: شماره step
            agent_id: شناسه agent
            action: action انجام شده
            reward: reward دریافتی
            info: اطلاعات اضافی
        """
        timestamp = datetime.now().isoformat()
        
        step_data = [
            episode, step, timestamp, agent_id,
            action, reward, info.get('value', 0), info.get('entropy', 0),
            info.get('task_id', -1), info.get('offload_target', ''),
            info.get('delay', 0), info.get('energy', 0)
        ]
        
        self.step_buffer.append(step_data)
        
        # لاگ GNN metrics
        if 'critical_scores' in info:
            gnn_data = [
                episode, step, timestamp,
                info.get('num_nodes', 0), info.get('num_edges', 0),
                np.mean(info['critical_scores']),
                np.max(info['critical_scores']),
                info.get('critical_path_length', 0),
                info.get('graph_density', 0)
            ]
            self.gnn_buffer.append(gnn_data)
    
    def log_episode(
        self,
        episode: int,
        total_reward: float,
        metrics: Dict
    ):
        """
        لاگ یک episode کامل
        
        Args:
            episode: شماره episode
            total_reward: مجموع reward
            metrics: metrics episode
        """
        timestamp = datetime.now().isoformat()
        
        episode_data = [
            episode, timestamp, total_reward, metrics.get('avg_reward', 0),
            metrics.get('actor_loss', 0), metrics.get('critic_loss', 0),
            metrics.get('entropy', 0), metrics.get('episode_length', 0),
            metrics.get('success_rate', 0), metrics.get('avg_delay', 0),
            metrics.get('avg_energy', 0), metrics.get('num_offloaded_tasks', 0)
        ]
        
        self.episode_buffer.append(episode_data)
        
        # flush buffer هر 10 episode
        if len(self.episode_buffer) >= 10:
            self.flush()
    
    def flush(self):
        """ذخیره buffer در فایل"""
        # Episode metrics
        if self.episode_buffer:
            with open(self.episode_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.episode_buffer)
            self.episode_buffer = []
        
        # Step metrics
        if self.step_buffer:
            with open(self.step_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.step_buffer)
            self.step_buffer = []
        
        # GNN metrics
        if self.gnn_buffer:
            with open(self.gnn_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.gnn_buffer)
            self.gnn_buffer = []
    
    def save_config(self, config: Dict):
        """ذخیره تنظیمات آموزش"""
        config_path = self.session_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Config saved to {config_path}")
    
    def get_latest_metrics(self, n: int = 100) -> Dict:
        """
        دریافت آخرین metrics برای نمایش realtime
        
        Args:
            n: تعداد آخرین record‌ها
            
        Returns:
            metrics: dictionary حاوی metrics
        """
        import pandas as pd
        
        try:
            # خواندن episode metrics
            df_ep = pd.read_csv(self.episode_log_path)
            latest_episodes = df_ep.tail(n)
            
            # خواندن GNN metrics
            df_gnn = pd.read_csv(self.gnn_log_path)
            latest_gnn = df_gnn.tail(n)
            
            return {
                'episodes': latest_episodes.to_dict('records'),
                'gnn_metrics': latest_gnn.to_dict('records'),
                'summary': {
                    'total_episodes': len(df_ep),
                    'avg_reward_last_100': latest_episodes['total_reward'].mean(),
                    'best_reward': df_ep['total_reward'].max(),
                    'avg_critical_score': latest_gnn['avg_critical_score'].mean()
                }
            }
        except Exception as e:
            print(f"⚠️ Error reading metrics: {e}")
            return {}
