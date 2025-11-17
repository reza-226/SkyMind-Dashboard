"""
core/collision_checker.py
===================================
الگوریتم‌های پیشرفته برای تشخیص برخورد و مسیریابی

Author: UAV Multi-Agent System
Version: 2.1 - Compatible with obstacles.py v2.1
"""

import numpy as np
from typing import List, Tuple, Optional
from core.obstacles import Obstacle, ObstacleManager


class CollisionChecker:
    """سیستم پیشرفته تشخیص برخورد"""
    
    def __init__(self, obstacle_manager: ObstacleManager, safety_margin: float = 2.0):
        self.obstacle_manager = obstacle_manager
        self.safety_margin = safety_margin
    
    def is_safe_position(
        self, 
        position: np.ndarray, 
        uav_radius: float = 1.0
    ) -> bool:
        """بررسی امنیت یک موقعیت"""
        # بررسی مرزهای نقشه
        if np.any(position < 0) or np.any(position > self.obstacle_manager.map_size):
            return False
        
        # بررسی برخورد با موانع
        for obstacle in self.obstacle_manager.obstacles:
            distance = np.linalg.norm(position - obstacle.position)
            min_safe_distance = obstacle.radius + uav_radius + self.safety_margin
            
            if distance < min_safe_distance:
                return False
        
        return True
    
    def is_safe_trajectory(
        self, 
        waypoints: List[np.ndarray],
        uav_radius: float = 1.0
    ) -> bool:
        """بررسی امنیت یک مسیر کامل"""
        for i in range(len(waypoints) - 1):
            collision, _ = self.check_path_collision(
                waypoints[i], 
                waypoints[i+1], 
                n_samples=30,
                uav_radius=uav_radius
            )
            if collision:
                return False
        return True
    
    def check_path_collision(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        n_samples: int = 10,
        uav_radius: float = 1.0
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """بررسی برخورد در مسیر"""
        for t in np.linspace(0, 1, n_samples):
            sample_pos = start_pos * (1 - t) + end_pos * t
            
            if not self.is_safe_position(sample_pos, uav_radius):
                return True, sample_pos
        
        return False, None
    
    def find_safe_path(
        self, 
        start: np.ndarray, 
        goal: np.ndarray, 
        max_iterations: int = 100,
        uav_radius: float = 1.0
    ) -> Optional[List[np.ndarray]]:
        """یافتن مسیر امن با الگوریتم RRT ساده"""
        # اگر مسیر مستقیم امن است
        collision, _ = self.check_path_collision(start, goal, uav_radius=uav_radius)
        if not collision:
            return [start, goal]
        
        # الگوریتم RRT ساده
        nodes = [start]
        parent = {tuple(start): None}
        
        for _ in range(max_iterations):
            # نمونه‌برداری تصادفی
            if np.random.rand() < 0.1:
                random_point = goal
            else:
                random_point = np.random.uniform(0, self.obstacle_manager.map_size, 2)
            
            # یافتن نزدیک‌ترین نود
            nearest_node = min(nodes, key=lambda n: np.linalg.norm(n - random_point))
            
            # گام به سمت نقطه تصادفی
            direction = random_point - nearest_node
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                step_size = min(10, distance)
                new_node = nearest_node + (direction / distance) * step_size
                
                # بررسی امنیت
                if self.is_safe_position(new_node, uav_radius):
                    collision, _ = self.check_path_collision(
                        nearest_node, new_node, uav_radius=uav_radius
                    )
                    if not collision:
                        nodes.append(new_node)
                        parent[tuple(new_node)] = nearest_node
                        
                        # بررسی رسیدن به هدف
                        if np.linalg.norm(new_node - goal) < step_size:
                            parent[tuple(goal)] = new_node
                            
                            # بازسازی مسیر
                            path = []
                            current = goal
                            while current is not None:
                                path.append(current)
                                current = parent.get(tuple(current))
                            
                            return list(reversed(path))
        
        return None
    
    def compute_collision_risk(
        self, 
        position: np.ndarray, 
        radius: float = 10.0,
        uav_radius: float = 1.0
    ) -> float:
        """
        محاسبه ریسک برخورد در یک موقعیت
        
        ✅ اصلاح‌شده: بدون نیاز به get_nearest_obstacles
        """
        if not self.obstacle_manager.obstacles:
            return 0.0
        
        # ✅ محاسبه فاصله از تمام موانع
        distances = []
        for obstacle in self.obstacle_manager.obstacles:
            distance = np.linalg.norm(position - obstacle.position)
            distances.append((distance, obstacle.radius))
        
        # ✅ مرتب‌سازی بر اساس فاصله
        distances.sort(key=lambda x: x[0])
        
        # ✅ انتخاب 5 مانع نزدیک
        nearest_obstacles = distances[:5]
        
        if not nearest_obstacles:
            return 0.0
        
        total_risk = 0.0
        
        for distance, obs_radius in nearest_obstacles:
            # برخورد مستقیم
            if distance < obs_radius + uav_radius + self.safety_margin:
                return 1.0
            
            # ریسک نمایی
            risk = np.exp(-(distance - obs_radius - uav_radius) / radius)
            total_risk += risk
        
        # نرمال‌سازی
        return np.clip(total_risk / len(nearest_obstacles), 0, 1)
    
    def get_avoidance_vector(
        self, 
        position: np.ndarray, 
        radius: float = 15.0
    ) -> np.ndarray:
        """محاسبه بردار اجتناب از موانع"""
        avoidance = np.zeros(2)
        
        for obstacle in self.obstacle_manager.obstacles:
            direction = position - obstacle.position
            distance = np.linalg.norm(direction)
            
            if distance < radius:
                if distance > 1e-6:
                    repulsion_strength = (radius - distance) / radius
                    avoidance += (direction / distance) * repulsion_strength
        
        # نرمال‌سازی
        norm = np.linalg.norm(avoidance)
        if norm > 1e-6:
            avoidance = avoidance / norm
        
        return avoidance
    
    def get_safe_direction(
        self,
        position: np.ndarray,
        desired_direction: np.ndarray,
        uav_radius: float = 1.0,
        look_ahead: float = 5.0
    ) -> np.ndarray:
        """یافتن جهت امن حرکت"""
        # نرمال‌سازی جهت مطلوب
        norm = np.linalg.norm(desired_direction)
        if norm > 1e-6:
            desired_direction = desired_direction / norm
        else:
            return np.zeros(2)
        
        # بررسی جهت مطلوب
        future_pos = position + desired_direction * look_ahead
        
        if self.is_safe_position(future_pos, uav_radius):
            return desired_direction
        
        # جستجوی جهت امن
        for angle in np.linspace(-np.pi, np.pi, 16):
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            test_direction = rotation_matrix @ desired_direction
            test_pos = position + test_direction * look_ahead
            
            if self.is_safe_position(test_pos, uav_radius):
                return test_direction
        
        return np.zeros(2)
    
    def get_collision_statistics(self) -> dict:
        """آمار برخوردها و تنظیمات"""
        return {
            'total_obstacles': len(self.obstacle_manager.obstacles),
            'safety_margin': self.safety_margin,
            'obstacle_mode': self.obstacle_manager.mode.value,
            'map_size': self.obstacle_manager.map_size
        }


# ==================== تست واحد ====================

if __name__ == "__main__":
    print("🧪 تست سیستم تشخیص برخورد پیشرفته")
    print("=" * 70)
    
    # ایجاد مدیر موانع
    manager = ObstacleManager(
        map_size=100.0, 
        obstacle_mode="complex",
        num_obstacles=15,  # ✅ استفاده از num_obstacles
        seed=42
    )
    
    checker = CollisionChecker(manager, safety_margin=3.0)
    
    print(f"\n📊 تنظیمات:")
    stats = checker.get_collision_statistics()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    # تست موقعیت‌های امن
    print(f"\n{'='*70}")
    print("📍 تست 1: بررسی موقعیت‌های امن")
    print("="*70)
    
    test_positions = [
        np.array([10.0, 10.0]),
        np.array([50.0, 50.0]),
        np.array([90.0, 90.0])
    ]
    
    for i, pos in enumerate(test_positions, 1):
        safe = checker.is_safe_position(pos)
        risk = checker.compute_collision_risk(pos, radius=15.0)
        avoidance = checker.get_avoidance_vector(pos)
        
        print(f"\n  موقعیت {i}: [{pos[0]:.1f}, {pos[1]:.1f}]")
        print(f"    ├─ امنیت: {'✅ امن' if safe else '❌ خطرناک'}")
        print(f"    ├─ ریسک: {risk:.1%}")
        print(f"    └─ بردار اجتناب: [{avoidance[0]:.2f}, {avoidance[1]:.2f}]")
    
    print(f"\n{'='*70}")
    print("✅ تست با موفقیت انجام شد!")
    print("="*70)
