"""
Obstacle Manager Wrapper
========================
واسط سازگاری برای استفاده در env_multi.py

این ماژول یک wrapper ساده برای ObstacleManager در obstacles.py است.
"""

from core.obstacles import ObstacleManager as CoreObstacleManager
from core.obstacles import ObstacleMode
import numpy as np


class ObstacleManager:
    """
    Wrapper برای سازگاری با env_multi.py
    
    این کلاس ObstacleManager از obstacles.py را wrap می‌کند
    تا با interface مورد انتظار env_multi.py سازگار شود.
    """
    
    def __init__(
        self,
        map_size: float = 1000.0,
        n_obstacles: int = 5,
        obstacle_mode: str = 'static',
        seed: int = None
    ):
        """
        Args:
            map_size: اندازه نقشه
            n_obstacles: تعداد موانع
            obstacle_mode: نوع موانع ('static', 'dynamic', 'mixed', 'complex')
            seed: seed برای تکرارپذیری
        """
        # تبدیل string به enum
        if obstacle_mode == 'static':
            mode = ObstacleMode.STATIC
        elif obstacle_mode == 'dynamic':
            mode = ObstacleMode.DYNAMIC
        elif obstacle_mode == 'mixed':
            mode = ObstacleMode.MIXED
        elif obstacle_mode == 'complex':
            mode = ObstacleMode.COMPLEX
        else:
            mode = ObstacleMode.STATIC
        
        # ایجاد ObstacleManager اصلی
        self.manager = CoreObstacleManager(
            map_size=map_size,
            n_obstacles=n_obstacles,
            obstacle_mode=mode,
            seed=seed
        )
        
        self.map_size = map_size
        self.n_obstacles = n_obstacles
        self.obstacle_mode = obstacle_mode
        self.seed = seed
    
    @property
    def obstacles(self):
        """دسترسی به لیست موانع"""
        return self.manager.obstacles
    
    def update(self):
        """به‌روزرسانی موانع"""
        self.manager.update()
    
    def reset(self):
        """بازنشانی موانع"""
        self.manager.reset()
    
    def get_nearest_obstacles(self, position: np.ndarray, n: int = 3):
        """یافتن نزدیک‌ترین موانع"""
        return self.manager.get_nearest_obstacles(position, n)


# Re-export برای سازگاری
__all__ = ['ObstacleManager']
