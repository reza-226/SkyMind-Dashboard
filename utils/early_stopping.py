"""
Early Stopping Monitor for MADDPG Training
سیستم توقف زودهنگام برای جلوگیری از Overfitting
"""

import numpy as np
from typing import Dict, Any, Optional
from collections import deque


class EarlyStopping:
    """
    کلاس نظارت بر توقف زودهنگام آموزش
    """
    
    def __init__(
        self,
        patience: int = 100,
        min_episodes: int = 200,
        metric_threshold: float = 0.1,
        min_delta: float = 0.001,
        mode: str = 'min',
    ):
        """
        Args:
            patience: تعداد اپیزودهایی که بدون بهبود صبر می‌کند
            min_episodes: حداقل تعداد اپیزودها قبل از فعال‌سازی
            metric_threshold: آستانه تغییرات متریک برای early stop
            min_delta: حداقل تغییر برای در نظر گرفتن بهبود
            mode: 'min' برای کمینه‌سازی یا 'max' برای بیشینه‌سازی
        """
        self.patience = patience
        self.min_episodes = min_episodes
        self.metric_threshold = metric_threshold
        self.min_delta = min_delta
        self.mode = mode
        
        # متغیرهای داخلی
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_episode = 0
        
        # تاریخچه برای تحلیل
        self.loss_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        # پیکربندی بر اساس mode
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.inf
    
    def check_health(self, metrics: Dict[str, Any]) -> bool:
        """
        بررسی سلامت آموزش و تصمیم‌گیری برای توقف
        
        Args:
            metrics: دیکشنری شامل 'episode' و 'critic_loss' (و سایر متریک‌ها)
            
        Returns:
            True اگر باید آموزش متوقف شود
        """
        episode = metrics.get('episode', 0)
        
        # منتظر حداقل تعداد اپیزود می‌مانیم
        if episode < self.min_episodes:
            return False
        
        # دریافت متریک اصلی (critic_loss)
        score = metrics.get('critic_loss', None)
        
        if score is None:
            return False
        
        # اضافه کردن به تاریخچه
        self.loss_history.append(score)
        
        if 'avg_reward' in metrics:
            self.reward_history.append(metrics['avg_reward'])
        
        # بررسی بهبود
        if self.monitor_op(score, self.best_score - self.min_delta):
            self.best_score = score
            self.best_episode = episode
            self.counter = 0
        else:
            self.counter += 1
        
        # بررسی شرایط توقف
        if self.counter >= self.patience:
            print(f"\n⚠️ Early Stopping: No improvement for {self.patience} episodes")
            print(f"   Best score: {self.best_score:.6f} at episode {self.best_episode}")
            self.early_stop = True
            return True
        
        # بررسی ناپایداری (اختیاری)
        if self._check_instability():
            print(f"\n⚠️ Training instability detected!")
            self.early_stop = True
            return True
        
        return False
    
    def _check_instability(self) -> bool:
        """
        بررسی ناپایداری در آموزش
        
        Returns:
            True اگر ناپایداری شدید مشاهده شود
        """
        if len(self.loss_history) < 50:
            return False
        
        recent_losses = list(self.loss_history)[-50:]
        
        # بررسی انفجار loss
        if np.mean(recent_losses) > 1e6:
            return True
        
        # بررسی NaN یا Inf
        if np.any(np.isnan(recent_losses)) or np.any(np.isinf(recent_losses)):
            return True
        
        # بررسی واریانس بیش از حد
        if np.std(recent_losses) > 100 * np.mean(recent_losses):
            return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی
        
        Returns:
            دیکشنری شامل اطلاعات وضعیت
        """
        return {
            'counter': self.counter,
            'patience': self.patience,
            'best_score': self.best_score,
            'best_episode': self.best_episode,
            'early_stop': self.early_stop,
            'avg_loss_50': np.mean(list(self.loss_history)[-50:]) if len(self.loss_history) > 0 else 0,
            'avg_reward_50': np.mean(list(self.reward_history)[-50:]) if len(self.reward_history) > 0 else 0,
        }
    
    def reset(self):
        """ریست کردن وضعیت"""
        self.counter = 0
        self.early_stop = False
        if self.mode == 'min':
            self.best_score = np.inf
        else:
            self.best_score = -np.inf
        self.best_episode = 0
        self.loss_history.clear()
        self.reward_history.clear()


class PerformanceMonitor:
    """
    کلاس نظارت بر عملکرد کلی آموزش
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: اندازه پنجره برای محاسبه میانگین‌های متحرک
        """
        self.window_size = window_size
        
        # تاریخچه متریک‌ها
        self.rewards = deque(maxlen=window_size)
        self.actor_losses = deque(maxlen=window_size)
        self.critic_losses = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        
        # آمار کلی
        self.total_episodes = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        self.worst_reward = float('inf')
    
    def update(self, metrics: Dict[str, Any]):
        """
        به‌روزرسانی با متریک‌های جدید
        
        Args:
            metrics: دیکشنری متریک‌های اپیزود
        """
        reward = metrics.get('avg_reward', 0)
        actor_loss = metrics.get('actor_loss', 0)
        critic_loss = metrics.get('critic_loss', 0)
        steps = metrics.get('steps', 0)
        
        self.rewards.append(reward)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.episode_lengths.append(steps)
        
        self.total_episodes += 1
        self.total_steps += steps
        self.best_reward = max(self.best_reward, reward)
        self.worst_reward = min(self.worst_reward, reward)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        دریافت آمار فعلی
        
        Returns:
            دیکشنری آمار
        """
        return {
            'reward_mean': np.mean(self.rewards) if len(self.rewards) > 0 else 0,
            'reward_std': np.std(self.rewards) if len(self.rewards) > 0 else 0,
            'reward_min': np.min(self.rewards) if len(self.rewards) > 0 else 0,
            'reward_max': np.max(self.rewards) if len(self.rewards) > 0 else 0,
            'actor_loss_mean': np.mean(self.actor_losses) if len(self.actor_losses) > 0 else 0,
            'critic_loss_mean': np.mean(self.critic_losses) if len(self.critic_losses) > 0 else 0,
            'avg_episode_length': np.mean(self.episode_lengths) if len(self.episode_lengths) > 0 else 0,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'best_reward_overall': self.best_reward,
            'worst_reward_overall': self.worst_reward,
        }
    
    def is_improving(self, threshold: float = 0.01) -> bool:
        """
        بررسی اینکه آیا عملکرد در حال بهبود است
        
        Args:
            threshold: آستانه بهبود
            
        Returns:
            True اگر بهبود مشاهده شود
        """
        if len(self.rewards) < self.window_size // 2:
            return True  # هنوز داده کافی نداریم
        
        recent_rewards = list(self.rewards)
        first_half = recent_rewards[:len(recent_rewards)//2]
        second_half = recent_rewards[len(recent_rewards)//2:]
        
        improvement = np.mean(second_half) - np.mean(first_half)
        
        return improvement > threshold
