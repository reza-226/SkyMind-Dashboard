"""
early_stopping_monitor.py
سیستم نظارت و توقف زودهنگام آموزش
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """تنظیمات آستانه برای هر متریک"""
    # Reward
    reward_critical: float = -50.0
    reward_warning: float = -30.0
    
    # Loss
    critic_loss_critical: float = 100.0
    critic_loss_warning: float = 50.0
    
    actor_loss_critical: float = 50.0
    actor_loss_warning: float = 25.0
    
    # Gradient
    grad_norm_critical: float = 10.0
    grad_norm_warning: float = 5.0
    
    # Action Saturation
    saturation_critical: float = 0.95
    saturation_warning: float = 0.85
    
    # Weight Drift
    weight_drift_critical: float = 2.0
    weight_drift_warning: float = 1.0


class EarlyStoppingMonitor:
    """
    سیستم پایش و توقف زودهنگام
    
    Features:
    - پایش متریک‌های کلیدی در پنجره‌های زمانی
    - تشخیص وضعیت Critical/Warning
    - توقف خودکار یا تعاملی
    - ذخیره گزارش‌ها به صورت JSON
    """
    
    def __init__(
        self,
        level: str,
        window_size: int = 100,
        check_interval: int = 100,
        auto_stop: bool = False,
        interactive: bool = True,
        save_dir: Optional[Path] = None,
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
        min_improvement: float = 0.01,
        patience: int = 5,
        min_episodes: int = 500  # ✅ حداقل تعداد episode قبل از چک کردن early stopping
    ):
        """
        Args:
            level: سطح آموزش (level1, level2, level3)
            window_size: تعداد episode برای میانگین‌گیری
            check_interval: هر چند episode بررسی شود
            auto_stop: آیا به صورت خودکار متوقف شود
            interactive: آیا از کاربر سؤال شود
            save_dir: پوشه ذخیره گزارش‌ها
            upper_bound: حد بالای reward برای توقف موفقیت‌آمیز (اختیاری)
            lower_bound: حد پایین reward برای توقف به دلیل عملکرد ضعیف (اختیاری)
            min_improvement: حداقل بهبود برای ریست patience
            patience: تعداد دفعات بدون بهبود قبل از توقف
            min_episodes: حداقل تعداد episode قبل از فعال شدن early stopping
        """
        self.level = level
        self.window_size = window_size
        self.check_interval = check_interval
        self.auto_stop = auto_stop
        self.interactive = interactive
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.min_improvement = min_improvement
        self.patience = patience
        self.min_episodes = min_episodes  # ✅
        
        # پوشه ذخیره
        if save_dir is None:
            save_dir = Path(f"models/{level}/monitoring")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # آستانه‌ها
        self.thresholds = self._get_level_thresholds()
        
        # تاریخچه
        self.history = {
            'rewards': [],
            'critic_losses': [],
            'actor_losses': [],
            'grad_norms': [],
            'action_saturations': [],
            'weight_drifts': []
        }
        
        # وضعیت
        self.consecutive_criticals = 0
        self.consecutive_warnings = 0
        self.total_checks = 0
        self.should_stop = False
        
        # برای پیگیری بهبود
        self.best_avg_reward = -float('inf')
        self.episodes_without_improvement = 0
        
        # گزارش‌ها
        self.check_reports = []
        
        logger.info(f"[MONITOR] Initialized for {level}")
        logger.info(f"  Window: {window_size}, Interval: {check_interval}")
        logger.info(f"  Auto-stop: {auto_stop}, Interactive: {interactive}")
        logger.info(f"  Min episodes before checking: {min_episodes}")  # ✅
        if upper_bound:
            logger.info(f"  Upper bound (success): {upper_bound}")
        if lower_bound:
            logger.info(f"  Lower bound (failure): {lower_bound}")
    
    def _get_level_thresholds(self) -> ThresholdConfig:
        """دریافت آستانه‌های مناسب برای هر سطح"""
        
        if self.level == "level1":
            return ThresholdConfig(
                reward_critical=-30.0,
                reward_warning=-20.0,
                critic_loss_critical=50.0,
                critic_loss_warning=30.0,
                saturation_critical=0.90,
                saturation_warning=0.80
            )
        
        elif self.level == "level2":
            return ThresholdConfig(
                reward_critical=-50.0,
                reward_warning=-35.0,
                critic_loss_critical=80.0,
                critic_loss_warning=50.0,
                saturation_critical=0.92,
                saturation_warning=0.82
            )
        
        else:  # level3
            return ThresholdConfig(
                reward_critical=-70.0,
                reward_warning=-50.0,
                critic_loss_critical=100.0,
                critic_loss_warning=60.0,
                saturation_critical=0.95,
                saturation_warning=0.85
            )
    
    def record_metrics(
        self,
        reward: float,
        critic_loss: Optional[float] = None,
        actor_loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        action_saturation: Optional[float] = None,
        weight_drift: Optional[float] = None
    ):
        """ثبت متریک‌های یک episode"""
        
        self.history['rewards'].append(reward)
        
        if critic_loss is not None:
            self.history['critic_losses'].append(critic_loss)
        
        if actor_loss is not None:
            self.history['actor_losses'].append(actor_loss)
        
        if grad_norm is not None:
            self.history['grad_norms'].append(grad_norm)
        
        if action_saturation is not None:
            self.history['action_saturations'].append(action_saturation)
        
        if weight_drift is not None:
            self.history['weight_drifts'].append(weight_drift)
    
    def check_health(self, episode: int) -> Dict:
        """
        بررسی سلامت آموزش
        
        Returns:
            گزارش وضعیت شامل:
            - status: 'healthy', 'warning', 'critical', 'success', 'failure'
            - issues: لیست مشکلات
            - should_stop: آیا باید متوقف شود
        """
        
        # ✅ اگر هنوز به min_episodes نرسیده‌ایم، چک نکن
        if episode < self.min_episodes:
            return {
                'status': 'healthy',
                'issues': [],
                'should_stop': False,
                'message': f'Skipping check - episode {episode} < min_episodes {self.min_episodes}'
            }
        
        if len(self.history['rewards']) < self.window_size:
            return {
                'status': 'healthy',
                'issues': [],
                'should_stop': False,
                'message': 'Not enough data for analysis'
            }
        
        # محاسبه میانگین‌ها
        window = slice(-self.window_size, None)
        
        metrics = {
            'mean_reward': sum(self.history['rewards'][window]) / self.window_size,
            'mean_critic_loss': (
                sum(self.history['critic_losses'][window]) / len(self.history['critic_losses'][window])
                if self.history['critic_losses'] else None
            ),
            'mean_actor_loss': (
                sum(self.history['actor_losses'][window]) / len(self.history['actor_losses'][window])
                if self.history['actor_losses'] else None
            ),
            'mean_grad_norm': (
                sum(self.history['grad_norms'][window]) / len(self.history['grad_norms'][window])
                if self.history['grad_norms'] else None
            ),
            'mean_saturation': (
                sum(self.history['action_saturations'][window]) / len(self.history['action_saturations'][window])
                if self.history['action_saturations'] else None
            ),
            'mean_weight_drift': (
                sum(self.history['weight_drifts'][window]) / len(self.history['weight_drifts'][window])
                if self.history['weight_drifts'] else None
            )
        }
        
        # چک کردن upper_bound (موفقیت)
        if self.upper_bound and metrics['mean_reward'] >= self.upper_bound:
            logger.info(f"🎯 Upper bound {self.upper_bound} reached! Mean reward: {metrics['mean_reward']:.2f}")
            self.should_stop = True
            return {
                'episode': episode,
                'status': 'success',
                'metrics': metrics,
                'issues': [],
                'should_stop': True,
                'stop_reason': f"Upper bound {self.upper_bound} achieved - Training successful!"
            }
        
        # چک کردن lower_bound (شکست)
        if self.lower_bound and metrics['mean_reward'] <= self.lower_bound:
            logger.warning(f"❌ Lower bound {self.lower_bound} reached! Mean reward: {metrics['mean_reward']:.2f}")
            self.should_stop = True
            return {
                'episode': episode,
                'status': 'failure',
                'metrics': metrics,
                'issues': [f"CRITICAL: Mean reward {metrics['mean_reward']:.2f} <= lower bound {self.lower_bound}"],
                'should_stop': True,
                'stop_reason': f"Lower bound {self.lower_bound} - Training failed"
            }
        
        # چک کردن بهبود
        if metrics['mean_reward'] > self.best_avg_reward + self.min_improvement:
            self.best_avg_reward = metrics['mean_reward']
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
        
        # بررسی آستانه‌ها
        issues = []
        critical_count = 0
        warning_count = 0
        
        # 1. Reward
        if metrics['mean_reward'] < self.thresholds.reward_critical:
            issues.append(f"CRITICAL: Reward={metrics['mean_reward']:.2f} < {self.thresholds.reward_critical}")
            critical_count += 1
        elif metrics['mean_reward'] < self.thresholds.reward_warning:
            issues.append(f"WARNING: Reward={metrics['mean_reward']:.2f} < {self.thresholds.reward_warning}")
            warning_count += 1
        
        # 2. Critic Loss
        if metrics['mean_critic_loss'] is not None:
            if metrics['mean_critic_loss'] > self.thresholds.critic_loss_critical:
                issues.append(f"CRITICAL: Critic Loss={metrics['mean_critic_loss']:.2f} > {self.thresholds.critic_loss_critical}")
                critical_count += 1
            elif metrics['mean_critic_loss'] > self.thresholds.critic_loss_warning:
                issues.append(f"WARNING: Critic Loss={metrics['mean_critic_loss']:.2f} > {self.thresholds.critic_loss_warning}")
                warning_count += 1
        
        # 3. Action Saturation
        if metrics['mean_saturation'] is not None:
            if metrics['mean_saturation'] > self.thresholds.saturation_critical:
                issues.append(f"CRITICAL: Saturation={metrics['mean_saturation']:.2%} > {self.thresholds.saturation_critical:.0%}")
                critical_count += 1
            elif metrics['mean_saturation'] > self.thresholds.saturation_warning:
                issues.append(f"WARNING: Saturation={metrics['mean_saturation']:.2%} > {self.thresholds.saturation_warning:.0%}")
                warning_count += 1
        
        # تعیین وضعیت
        if critical_count > 0:
            status = 'critical'
            self.consecutive_criticals += 1
            self.consecutive_warnings = 0
        elif warning_count > 0:
            status = 'warning'
            self.consecutive_warnings += 1
            self.consecutive_criticals = 0
        else:
            status = 'healthy'
            self.consecutive_criticals = 0
            self.consecutive_warnings = 0
        
        # تصمیم به توقف
        should_stop = False
        stop_reason = None
        
        if self.consecutive_criticals >= 2:
            should_stop = True
            stop_reason = "2 consecutive critical checks"
        
        elif self.consecutive_criticals >= 1 and self.consecutive_warnings >= 2:
            should_stop = True
            stop_reason = "1 critical + 2 warnings"
        
        # چک patience
        if self.episodes_without_improvement >= self.patience:
            should_stop = True
            stop_reason = f"No improvement for {self.patience} checks"
        
        # گزارش
        report = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'metrics': metrics,
            'issues': issues,
            'consecutive_criticals': self.consecutive_criticals,
            'consecutive_warnings': self.consecutive_warnings,
            'episodes_without_improvement': self.episodes_without_improvement,
            'should_stop': should_stop,
            'stop_reason': stop_reason
        }
        
        self.check_reports.append(report)
        self.total_checks += 1
        
        # ذخیره گزارش
        self._save_check_report(episode, report)
        
        # نمایش
        if issues:
            logger.warning(f"\n⚠️ Health Check #{self.total_checks} (Episode {episode}):")
            logger.warning(f"   Status: {status.upper()}")
            for issue in issues:
                logger.warning(f"   - {issue}")
        
        # تعامل با کاربر
        if should_stop:
            self.should_stop = self._handle_stop_decision(report)
        
        return report
    
    def _handle_stop_decision(self, report: Dict) -> bool:
        """مدیریت تصمیم توقف"""
        
        logger.critical("\n" + "="*80)
        logger.critical("🚨 EARLY STOPPING TRIGGERED!")
        logger.critical("="*80)
        logger.critical(f"Reason: {report['stop_reason']}")
        logger.critical(f"Status: {report['status'].upper()}")
        
        if self.auto_stop:
            logger.critical("🛑 Auto-stopping enabled. Training will halt.")
            return True
        
        if self.interactive:
            logger.critical("\nOptions:")
            logger.critical("  [1] Stop training immediately")
            logger.critical("  [2] Continue for 100 more episodes")
            logger.critical("  [3] Ignore and continue")
            
            try:
                choice = input("\nYour choice [1/2/3]: ").strip()
                
                if choice == '1':
                    logger.critical("✅ Stopping training...")
                    return True
                elif choice == '2':
                    logger.critical("⏳ Continuing for 100 more episodes...")
                    self.consecutive_criticals = 0
                    self.consecutive_warnings = 0
                    self.episodes_without_improvement = 0
                    return False
                else:
                    logger.critical("▶️ Ignoring warning and continuing...")
                    self.consecutive_criticals = 0
                    self.consecutive_warnings = 0
                    self.episodes_without_improvement = 0
                    return False
            
            except KeyboardInterrupt:
                logger.critical("\n🛑 User interrupted. Stopping...")
                return True
        
        else:
            logger.critical("⚠️ Early stopping condition met, but non-interactive mode.")
            logger.critical("   Training continues. Set auto_stop=True or interactive=True to control.")
            return False
    
    def _save_check_report(self, episode: int, report: Dict):
        """ذخیره گزارش بررسی"""
        
        filename = self.save_dir / f"check_ep{episode}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
    
    def save_final_report(self):
        """ذخیره گزارش نهایی"""
        
        report = {
            'level': self.level,
            'total_episodes': len(self.history['rewards']),
            'total_checks': self.total_checks,
            'final_status': 'stopped' if self.should_stop else 'completed',
            'best_avg_reward': self.best_avg_reward,
            'all_checks': self.check_reports,
            'thresholds': asdict(self.thresholds),
            'summary': {
                'total_criticals': sum(1 for r in self.check_reports if r['status'] == 'critical'),
                'total_warnings': sum(1 for r in self.check_reports if r['status'] == 'warning'),
                'total_healthy': sum(1 for r in self.check_reports if r['status'] == 'healthy')
            }
        }
        
        filename = self.save_dir / f"final_report_{self.level}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📊 Final report saved: {filename}")
