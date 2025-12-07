"""
Metrics Collector for GE-CL-MADDPG Training
جمع‌آوری و مدیریت متریک‌های آموزش
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict, deque


class MetricsCollector:
    """
    سیستم جامع جمع‌آوری متریک‌های آموزش
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: اندازه پنجره برای میانگین متحرک
        """
        self.window_size = window_size
        
        # ذخیره‌سازی متریک‌ها
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        
        # متریک‌های تخصصی UAV-MEC
        self.energy_consumption = []
        self.task_latencies = []
        self.success_rates = []
        self.collision_counts = []
        self.gini_indices = []
        self.collaboration_scores = []
        
        # پنجره‌های متحرک
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_success = deque(maxlen=window_size)
        
        # متریک‌های per-agent
        self.agent_metrics = defaultdict(lambda: {
            'rewards': [],
            'actions': [],
            'losses': []
        })
        
    def record_episode(self, 
                      episode: int,
                      total_reward: float,
                      episode_length: int,
                      actor_loss: float = 0.0,
                      critic_loss: float = 0.0,
                      **kwargs):
        """
        ثبت متریک‌های یک اپیزود
        
        Args:
            episode: شماره اپیزود
            total_reward: مجموع reward
            episode_length: طول اپیزود
            actor_loss: loss شبکه actor
            critic_loss: loss شبکه critic
            **kwargs: متریک‌های اضافی
        """
        # متریک‌های پایه
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        
        # پنجره متحرک
        self.recent_rewards.append(total_reward)
        
        # متریک‌های تخصصی
        if 'energy' in kwargs:
            self.energy_consumption.append(kwargs['energy'])
        if 'latency' in kwargs:
            self.task_latencies.append(kwargs['latency'])
        if 'success_rate' in kwargs:
            self.success_rates.append(kwargs['success_rate'])
            self.recent_success.append(kwargs['success_rate'])
        if 'collisions' in kwargs:
            self.collision_counts.append(kwargs['collisions'])
        if 'gini_index' in kwargs:
            self.gini_indices.append(kwargs['gini_index'])
        if 'collaboration_score' in kwargs:
            self.collaboration_scores.append(kwargs['collaboration_score'])
    
    def record_agent_step(self, agent_id: int, reward: float, 
                         action: np.ndarray, loss: float = 0.0):
        """
        ثبت متریک‌های یک گام برای یک agent
        
        Args:
            agent_id: شناسه agent
            reward: reward دریافتی
            action: عمل انجام شده
            loss: loss شبکه
        """
        self.agent_metrics[agent_id]['rewards'].append(reward)
        self.agent_metrics[agent_id]['actions'].append(action)
        self.agent_metrics[agent_id]['losses'].append(loss)
    
    def get_recent_avg_reward(self, window: Optional[int] = None) -> float:
        """
        محاسبه میانگین reward اخیر
        
        Args:
            window: اندازه پنجره (اگر None باشد از window_size استفاده می‌شود)
        
        Returns:
            میانگین reward
        """
        if not self.episode_rewards:
            return 0.0
        
        if window is None:
            return np.mean(list(self.recent_rewards))
        
        window = min(window, len(self.episode_rewards))
        return np.mean(self.episode_rewards[-window:])
    
    def get_recent_success_rate(self, window: Optional[int] = None) -> float:
        """
        محاسبه نرخ موفقیت اخیر
        
        Args:
            window: اندازه پنجره
        
        Returns:
            نرخ موفقیت
        """
        if not self.success_rates:
            return 0.0
        
        if window is None:
            return np.mean(list(self.recent_success))
        
        window = min(window, len(self.success_rates))
        return np.mean(self.success_rates[-window:])
    
    def get_summary(self) -> Dict:
        """
        دریافت خلاصه آماری
        
        Returns:
            دیکشنری شامل آمار کلیدی
        """
        if not self.episode_rewards:
            return {}
        
        summary = {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'best_reward': np.max(self.episode_rewards),
            'worst_reward': np.min(self.episode_rewards),
            'recent_avg_reward': self.get_recent_avg_reward(),
            'avg_episode_length': np.mean(self.episode_lengths),
        }
        
        # اضافه کردن متریک‌های تخصصی
        if self.energy_consumption:
            summary['avg_energy'] = np.mean(self.energy_consumption)
            summary['total_energy'] = np.sum(self.energy_consumption)
        
        if self.task_latencies:
            summary['avg_latency'] = np.mean(self.task_latencies)
            summary['min_latency'] = np.min(self.task_latencies)
        
        if self.success_rates:
            summary['avg_success_rate'] = np.mean(self.success_rates)
            summary['recent_success_rate'] = self.get_recent_success_rate()
        
        if self.collision_counts:
            summary['total_collisions'] = np.sum(self.collision_counts)
            summary['avg_collisions'] = np.mean(self.collision_counts)
        
        if self.gini_indices:
            summary['avg_gini_index'] = np.mean(self.gini_indices)
        
        if self.collaboration_scores:
            summary['avg_collaboration'] = np.mean(self.collaboration_scores)
        
        return summary
    
    def get_agent_summary(self, agent_id: int) -> Dict:
        """
        دریافت خلاصه متریک‌های یک agent خاص
        
        Args:
            agent_id: شناسه agent
        
        Returns:
            دیکشنری آمار agent
        """
        if agent_id not in self.agent_metrics:
            return {}
        
        metrics = self.agent_metrics[agent_id]
        
        return {
            'total_steps': len(metrics['rewards']),
            'avg_reward': np.mean(metrics['rewards']) if metrics['rewards'] else 0.0,
            'avg_loss': np.mean(metrics['losses']) if metrics['losses'] else 0.0,
            'total_reward': np.sum(metrics['rewards']) if metrics['rewards'] else 0.0,
        }
    
    def print_summary(self, episode: Optional[int] = None):
        """
        چاپ خلاصه آماری
        
        Args:
            episode: شماره اپیزود فعلی (اختیاری)
        """
        summary = self.get_summary()
        
        if not summary:
            print("⚠️ No metrics recorded yet")
            return
        
        print("\n" + "="*60)
        if episode is not None:
            print(f"📊 METRICS SUMMARY (Episode {episode})")
        else:
            print("📊 METRICS SUMMARY")
        print("="*60)
        
        print(f"Total Episodes: {summary['total_episodes']}")
        print(f"Average Reward: {summary['avg_reward']:.4f} ± {summary['std_reward']:.4f}")
        print(f"Recent Avg (100): {summary['recent_avg_reward']:.4f}")
        print(f"Best Reward: {summary['best_reward']:.4f}")
        print(f"Avg Episode Length: {summary['avg_episode_length']:.2f}")
        
        if 'avg_success_rate' in summary:
            print(f"Success Rate: {summary['avg_success_rate']*100:.2f}%")
        
        if 'avg_energy' in summary:
            print(f"Avg Energy: {summary['avg_energy']:.2f} J")
        
        if 'avg_latency' in summary:
            print(f"Avg Latency: {summary['avg_latency']:.2f} ms")
        
        if 'total_collisions' in summary:
            print(f"Total Collisions: {summary['total_collisions']}")
        
        print("="*60 + "\n")
    
    def reset(self):
        """ریست کردن تمام متریک‌ها"""
        self.__init__(window_size=self.window_size)


# ========================================
# Environment Metrics Collector
# ========================================

class EnvironmentMetricsCollector:
    """
    جمع‌آوری متریک‌های محیط در طول اجرا
    """
    
    def __init__(self):
        self.step_metrics = []
        self.episode_start_time = None
        
    def record_step(self, step: int, **metrics):
        """ثبت متریک‌های یک گام"""
        metrics['step'] = step
        self.step_metrics.append(metrics)
    
    def get_episode_metrics(self) -> Dict:
        """دریافت متریک‌های کل اپیزود"""
        if not self.step_metrics:
            return {}
        
        # تجمیع متریک‌ها
        aggregated = {
            'total_steps': len(self.step_metrics),
        }
        
        # میانگین‌گیری
        keys = set().union(*[m.keys() for m in self.step_metrics])
        keys.discard('step')
        
        for key in keys:
            values = [m.get(key, 0) for m in self.step_metrics]
            aggregated[f'avg_{key}'] = np.mean(values)
            aggregated[f'sum_{key}'] = np.sum(values)
            aggregated[f'max_{key}'] = np.max(values)
        
        return aggregated
    
    def reset(self):
        """ریست برای اپیزود جدید"""
        self.step_metrics = []
