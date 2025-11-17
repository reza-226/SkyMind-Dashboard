"""
مدیریت موانع در محیط شبیه‌سازی
Version: 2.1 - Compatible with train_maddpg_complete.py
"""
import numpy as np
from typing import List, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass


class ObstacleMode(Enum):
    """حالت‌های مختلف موانع"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    COMPLEX = "complex"
    NONE = "none"
    RANDOM = "random"


@dataclass
class Obstacle:
    """کلاس نمایش یک مانع"""
    position: np.ndarray
    radius: float
    velocity: np.ndarray = None
    
    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros(2)
    
    def is_collision(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:
        """بررسی برخورد یک نقطه با مانع"""
        distance = np.linalg.norm(point - self.position)
        return distance < (self.radius + safety_margin)


class ObstacleManager:
    """مدیریت موانع در محیط"""
    
    def __init__(
        self,
        map_size: float,
        obstacle_mode: Union[str, ObstacleMode] = "static",
        n_obstacles: int = 10,
        num_obstacles: int = None,  # ✅ پارامتر جدید
        min_radius: float = 2.0,    # ✅ پارامتر جدید
        max_radius: float = 5.0,    # ✅ پارامتر جدید
        seed: Optional[int] = None
    ):
        """
        Args:
            map_size: اندازه نقشه
            obstacle_mode: حالت موانع
            n_obstacles: تعداد موانع (نام قدیمی)
            num_obstacles: تعداد موانع (نام جدید)
            min_radius: حداقل شعاع موانع
            max_radius: حداکثر شعاع موانع
            seed: seed تصادفی
        """
        self.map_size = map_size
        
        # ✅ پشتیبانی از هر دو نام پارامتر
        if num_obstacles is not None:
            self.n_obstacles = num_obstacles
        else:
            self.n_obstacles = n_obstacles
        
        # ✅ ذخیره محدوده شعاع
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # تبدیل string به enum
        if isinstance(obstacle_mode, str):
            mode_map = {
                "static": ObstacleMode.STATIC,
                "dynamic": ObstacleMode.DYNAMIC,
                "complex": ObstacleMode.COMPLEX,
                "none": ObstacleMode.NONE,
                "random": ObstacleMode.RANDOM,
                "moderate": ObstacleMode.STATIC
            }
            self.mode = mode_map.get(obstacle_mode.lower(), ObstacleMode.STATIC)
        else:
            self.mode = obstacle_mode
        
        self.obstacles: List[Obstacle] = []
        
        # RNG برای تکرارپذیری
        self.rng = np.random.RandomState(seed)
        
        # ایجاد موانع اولیه
        if self.mode != ObstacleMode.NONE:
            self._generate_obstacles()
    
    def _generate_obstacles(self):
        """تولید موانع تصادفی"""
        self.obstacles.clear()
        
        for _ in range(self.n_obstacles):
            # موقعیت تصادفی
            position = self.rng.uniform(10, self.map_size - 10, 2)
            
            # ✅ استفاده از محدوده شعاع
            radius = self.rng.uniform(self.min_radius, self.max_radius)
            
            # تعیین سرعت
            if self.mode in [ObstacleMode.DYNAMIC, ObstacleMode.COMPLEX, ObstacleMode.RANDOM]:
                velocity = self.rng.uniform(-1, 1, 2)
            else:
                velocity = np.zeros(2)
            
            obstacle = Obstacle(
                position=position,
                radius=radius,
                velocity=velocity
            )
            self.obstacles.append(obstacle)
    
    def update(self, dt: float = 0.1):
        """به‌روزرسانی موانع پویا"""
        if self.mode in [ObstacleMode.DYNAMIC, ObstacleMode.COMPLEX, ObstacleMode.RANDOM]:
            for obstacle in self.obstacles:
                # حرکت مانع
                obstacle.position += obstacle.velocity * dt
                
                # بازگشت از مرزها
                for i in range(2):
                    if obstacle.position[i] < 0 or obstacle.position[i] > self.map_size:
                        obstacle.velocity[i] *= -1
                        obstacle.position[i] = np.clip(
                            obstacle.position[i], 0, self.map_size
                        )
    
    def reset(self):
        """بازنشانی موانع"""
        if self.mode != ObstacleMode.NONE:
            self._generate_obstacles()
    
    def get_obstacles_info(self) -> List[Tuple[np.ndarray, float]]:
        """دریافت اطلاعات موانع"""
        return [(obs.position.copy(), obs.radius) for obs in self.obstacles]
    
    def __len__(self) -> int:
        """تعداد موانع"""
        return len(self.obstacles)
    
    def __repr__(self) -> str:
        return f"ObstacleManager(mode={self.mode.value}, n_obstacles={self.n_obstacles})"


# ==================== تست ====================

if __name__ == "__main__":
    print("🧪 تست ObstacleManager")
    print("=" * 50)
    
    # تست با num_obstacles
    manager = ObstacleManager(
        map_size=100,
        obstacle_mode="static",
        num_obstacles=5,
        min_radius=2.0,
        max_radius=5.0,
        seed=42
    )
    
    print(f"\n✅ ایجاد شد: {manager}")
    print(f"تعداد موانع: {len(manager)}")
    
    for i, (pos, radius) in enumerate(manager.get_obstacles_info()):
        print(f"  مانع {i+1}: pos=[{pos[0]:.1f}, {pos[1]:.1f}], r={radius:.1f}")
    
    print("\n" + "=" * 50)
    print("✅ تست موفق!")
