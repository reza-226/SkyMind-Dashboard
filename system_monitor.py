# system_monitor.py
"""
System Monitoring for MADDPG Training
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class ThresholdConfig:
    """Configuration for threshold monitoring"""
    metric_name: str = "general_metric"
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    window_size: int = 100
    check_interval: int = 50
    patience: int = 10
    min_episodes: int = 200
    
    # آرگومان‌های اضافی که در Level 1, 2, 3 استفاده می‌شوند
    critic_loss_critical: Optional[float] = None
    saturation_critical: Optional[float] = None
    actor_loss_critical: Optional[float] = None
    drift_critic_critical: Optional[float] = None
    success_rate_critical: Optional[float] = None
    
    def __post_init__(self):
        """Set bounds based on critical values if provided"""
        if self.critic_loss_critical is not None and self.upper_bound is None:
            self.upper_bound = self.critic_loss_critical
        if self.saturation_critical is not None and self.upper_bound is None:
            self.upper_bound = self.saturation_critical
        if self.actor_loss_critical is not None and self.upper_bound is None:
            self.upper_bound = self.actor_loss_critical
        if self.drift_critic_critical is not None and self.upper_bound is None:
            self.upper_bound = self.drift_critic_critical
        if self.success_rate_critical is not None and self.lower_bound is None:
            self.lower_bound = self.success_rate_critical


class SystemMonitor:
    """
    Monitors system metrics and triggers emergency stops
    """
    
    def __init__(self, config: ThresholdConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.metric_history = []
        self.violation_count = 0
        self.total_checks = 0
        self.emergency_triggered = False
    
    def update(self, metric_value: float, episode: int) -> bool:
        """
        Update metric and check for violations
        
        Args:
            metric_value: Current metric value
            episode: Current episode number
            
        Returns:
            True if emergency stop triggered, False otherwise
        """
        self.metric_history.append(metric_value)
        
        # Keep only window_size recent values
        if len(self.metric_history) > self.config.window_size:
            self.metric_history.pop(0)
        
        # Don't check until minimum episodes reached
        if episode < self.config.min_episodes:
            return False
        
        # Only check at specified intervals
        if episode % self.config.check_interval != 0:
            return False
        
        self.total_checks += 1
        
        # Check thresholds
        if len(self.metric_history) >= self.config.window_size:
            avg_value = np.mean(self.metric_history[-self.config.window_size:])
            
            violation = False
            
            if self.config.lower_bound is not None:
                if avg_value < self.config.lower_bound:
                    violation = True
                    self.logger.warning(
                        f"⚠️ {self.config.metric_name} below threshold: "
                        f"{avg_value:.4f} < {self.config.lower_bound}"
                    )
            
            if self.config.upper_bound is not None:
                if avg_value > self.config.upper_bound:
                    violation = True
                    self.logger.warning(
                        f"⚠️ {self.config.metric_name} above threshold: "
                        f"{avg_value:.4f} > {self.config.upper_bound}"
                    )
            
            if violation:
                self.violation_count += 1
                
                if self.violation_count >= self.config.patience:
                    self.emergency_triggered = True
                    self.logger.error(
                        f"🚨 EMERGENCY STOP: {self.config.metric_name} "
                        f"violated thresholds {self.violation_count} times"
                    )
                    return True
            else:
                # Decay violation count if no violation
                self.violation_count = max(0, self.violation_count - 1)
        
        return False
    
    def get_status(self) -> Dict:
        """Get current monitoring status"""
        if len(self.metric_history) == 0:
            avg = None
        else:
            avg = np.mean(self.metric_history[-self.config.window_size:])
        
        return {
            'metric_name': self.config.metric_name,
            'current_avg': avg,
            'violation_count': self.violation_count,
            'total_checks': self.total_checks,
            'emergency_triggered': self.emergency_triggered
        }
    
    def reset(self):
        """Reset monitoring state"""
        self.metric_history = []
        self.violation_count = 0
        self.total_checks = 0
        self.emergency_triggered = False
