# core/env_adapter.py - فقط این فایل را بسازید
"""
Adapter بدون تغییر در env_multi.py موجود
"""
from typing import Dict, Any
import numpy as np

class EnvironmentAdapter:
    """
    این adapter با env موجود (بدون موانع) کار می‌کند
    و آینده‌نگر است
    """
    
    def __init__(self, env):
        self.env = env
        self.version = "1.0.0"  # نسخه فعلی بدون موانع
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        ✅ فقط داده‌های موجود فعلی را format می‌کند
        هیچ چیز جدیدی اضافه نمی‌شود
        """
        return {
            'uav_positions': self._get_uav_positions(),
            'user_positions': self._get_user_positions(),
            'obstacles': None,  # ✅ آماده برای آینده
            'metrics': self._get_metrics(),
            'timestamp': self._get_timestamp(),
            'version': self.version
        }
    
    def _get_uav_positions(self):
        """استخراج از env موجود - بدون تغییر"""
        positions = []
        for i in range(self.env.n_uavs):
            pos = self.env.uav_positions[i]
            positions.append({
                'id': i,
                'x': float(pos[0]),
                'y': float(pos[1]),
                'z': 0.0  # فعلاً 2D
            })
        return positions
    
    def _get_user_positions(self):
        """استخراج از env موجود - بدون تغییر"""
        positions = []
        for i in range(self.env.n_users):
            pos = self.env.user_positions[i]
            positions.append({
                'id': i,
                'x': float(pos[0]),
                'y': float(pos[1])
            })
        return positions
    
    def _get_metrics(self):
        """استخراج metrics از info dict موجود"""
        return {
            'reward': float(getattr(self.env, 'last_total_reward', 0.0)),
            'collision_count': 0,  # فعلاً صفر
            'coverage': self._calculate_coverage()
        }
    
    def _calculate_coverage(self) -> float:
        """محاسبه coverage - مثل قبل"""
        covered = 0
        for user_pos in self.env.user_positions:
            for uav_pos in self.env.uav_positions:
                dist = np.linalg.norm(user_pos - uav_pos)
                if dist < self.env.comm_range:
                    covered += 1
                    break
        return covered / self.env.n_users if self.env.n_users > 0 else 0.0
    
    def _get_timestamp(self):
        import time
        return int(time.time() * 1000)
