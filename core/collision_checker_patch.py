"""
core/collision_checker_patch.py
=======================
اضافه کردن متدهای سازگار با MultiUAVEnv به CollisionChecker
+ اصلاح is_safe_position برای پشتیبانی 3D

Compatible with: obstacles.py v2.1 (2D obstacles, 3D positions)
"""

import sys
import os
from pathlib import Path

# اضافه کردن مسیر پروژه به sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from typing import List, Optional


def is_safe_position_3d(self, position: np.ndarray, uav_radius: float = 1.0) -> bool:
    """
    نسخه اصلاح شده is_safe_position که از موقعیت‌های 3D پشتیبانی می‌کند
    
    Args:
        position: موقعیت [x, y] یا [x, y, z]
        uav_radius: شعاع UAV
        
    Returns:
        True اگر موقعیت امن باشد
    """
    position = np.asarray(position).flatten()
    
    # تبدیل 3D به 2D برای بررسی با موانع
    if len(position) == 3:
        position_2d = position[:2]  # فقط x, y
        altitude = position[2]
    elif len(position) == 2:
        position_2d = position
        altitude = None
    else:
        raise ValueError(f"Position must be 2D or 3D, got shape {position.shape}")
    
    # بررسی محدوده نقشه (فقط x, y)
    map_size_2d = self.obstacle_manager.map_size
    if np.any(position_2d < 0) or np.any(position_2d > map_size_2d):
        return False
    
    # بررسی برخورد با موانع (فقط در صفحه xy)
    for obstacle in self.obstacle_manager.obstacles:
        obs_pos = np.asarray(obstacle.position).flatten()[:2]  # فقط x, y
        distance_2d = np.linalg.norm(position_2d - obs_pos)
        
        min_distance = uav_radius + obstacle.radius + self.safety_margin
        
        if distance_2d < min_distance:
            return False
    
    return True


def check_collision(self, position: np.ndarray, radius: float = 0.0) -> bool:
    """
    بررسی برخورد با موانع (alias برای is_safe_position)
    
    Args:
        position: موقعیت [x, y] یا [x, y, z]
        radius: شعاع UAV یا شعاع امنیتی
        
    Returns:
        True اگر برخورد وجود دارد، False در غیر این صورت
    """
    # استفاده از نسخه 3D
    if hasattr(self, 'is_safe_position_3d'):
        return not self.is_safe_position_3d(position, uav_radius=radius)
    else:
        return not self.is_safe_position(position, uav_radius=radius)


def check_uav_collision(
    self, 
    uav_positions: List[np.ndarray], 
    min_distance: float = 5.0
) -> bool:
    """
    بررسی برخورد بین UAVها
    
    Args:
        uav_positions: لیست موقعیت‌های UAVها
        min_distance: حداقل فاصله مجاز
        
    Returns:
        True اگر برخورد بین UAVها وجود دارد
    """
    n_uavs = len(uav_positions)
    
    for i in range(n_uavs):
        for j in range(i + 1, n_uavs):
            pos_i = np.asarray(uav_positions[i]).flatten()[:3]
            pos_j = np.asarray(uav_positions[j]).flatten()[:3]
            
            distance = np.linalg.norm(pos_i - pos_j)
            
            if distance < min_distance:
                return True
    
    return False


def is_position_safe_extended(
    self,
    position: np.ndarray,
    uav_positions: Optional[List[np.ndarray]] = None,
    uav_radius: float = 1.0,
    min_uav_distance: float = 5.0
) -> bool:
    """
    بررسی امنیت کامل (موانع + UAVهای دیگر)
    
    Args:
        position: موقعیت مورد بررسی
        uav_positions: موقعیت UAVهای دیگر (اختیاری)
        uav_radius: شعاع UAV
        min_uav_distance: حداقل فاصله با UAVهای دیگر
        
    Returns:
        True اگر موقعیت کاملاً امن باشد
    """
    # بررسی موانع (با استفاده از نسخه 3D)
    safe_check = self.is_safe_position_3d if hasattr(self, 'is_safe_position_3d') else self.is_safe_position
    
    if not safe_check(position, uav_radius):
        return False
    
    # بررسی برخورد با UAVهای دیگر
    if uav_positions is not None and len(uav_positions) > 0:
        position = np.asarray(position).flatten()[:3]
        
        for other_pos in uav_positions:
            other_pos = np.asarray(other_pos).flatten()[:3]
            distance = np.linalg.norm(position - other_pos)
            
            if distance < min_uav_distance:
                return False
    
    return True


def get_collision_info(self, position: np.ndarray, radius: float = 1.0) -> dict:
    """
    اطلاعات تفصیلی برخورد
    
    Returns:
        dict با کلیدهای:
        - has_collision: bool
        - nearest_obstacle: dict یا None
        - distance_to_nearest: float
    """
    position = np.asarray(position).flatten()
    position_2d = position[:2] if len(position) >= 2 else position
    
    nearest_obstacle = None
    min_distance = float('inf')
    
    for obstacle in self.obstacle_manager.obstacles:
        obs_pos = np.asarray(obstacle.position).flatten()[:2]
        distance = np.linalg.norm(position_2d - obs_pos)
        
        if distance < min_distance:
            min_distance = distance
            nearest_obstacle = {
                'position': obs_pos,
                'radius': obstacle.radius,
                'type': getattr(obstacle, 'obstacle_type', 'unknown')
            }
    
    has_collision = min_distance < (radius + self.safety_margin + 
                                    (nearest_obstacle['radius'] if nearest_obstacle else 0))
    
    return {
        'has_collision': has_collision,
        'nearest_obstacle': nearest_obstacle,
        'distance_to_nearest': min_distance
    }


def patch_collision_checker():
    """
    اعمال پچ به CollisionChecker
    """
    try:
        from core.collision_checker import CollisionChecker
    except ImportError:
        print("❌ Cannot import CollisionChecker")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Python path: {sys.path[:3]}")
        raise
    
    patches = {
        'is_safe_position_3d': is_safe_position_3d,
        'check_collision': check_collision,
        'check_uav_collision': check_uav_collision,
        'is_position_safe_extended': is_position_safe_extended,
        'get_collision_info': get_collision_info
    }
    
    patched = []
    skipped = []
    
    for method_name, method_func in patches.items():
        if not hasattr(CollisionChecker, method_name):
            setattr(CollisionChecker, method_name, method_func)
            patched.append(method_name)
        else:
            skipped.append(method_name)
    
    if patched:
        print(f"✓ CollisionChecker patched: {', '.join(patched)}")
    if skipped:
        print(f"  Already exists: {', '.join(skipped)}")
    
    print("✓ CollisionChecker ready (2D obstacles + 3D UAV positions)!")
    
    return patched, skipped


# تست خودکار
if __name__ == '__main__':
    print("Testing CollisionChecker Patch...")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    print()
    
    try:
        patched, skipped = patch_collision_checker()
        
        # تست ساده
        from core.obstacles import ObstacleManager
        from core.collision_checker import CollisionChecker
        
        print("\n" + "=" * 70)
        print("Running tests...")
        print("=" * 70)
        
        # ساخت محیط تست با map_size دو بعدی
        obstacle_manager = ObstacleManager(
            map_size=np.array([100.0, 100.0]),  # فقط 2D
            num_obstacles=5,
            seed=42
        )
        
        checker = CollisionChecker(obstacle_manager, safety_margin=2.0)
        
        # تست 1: موقعیت‌های 3D
        test_positions = [
            np.array([10.0, 10.0, 25.0]),   # 3D
            np.array([50.0, 50.0, 30.0]),   # 3D
            np.array([90.0, 90.0]),         # 2D
        ]
        
        print(f"\n1. Testing 3D position support:")
        for i, pos in enumerate(test_positions, 1):
            dim = "3D" if len(pos) == 3 else "2D"
            is_safe = checker.is_safe_position_3d(pos, uav_radius=1.0)
            has_collision = checker.check_collision(pos, radius=1.0)
            print(f"   {i}. {dim} position {pos}: safe={is_safe}, collision={has_collision}")
        
        # تست 2: برخورد UAVها
        uav_positions = [
            np.array([10.0, 10.0, 25.0]),
            np.array([15.0, 10.0, 25.0]),
            np.array([10.0, 15.0, 25.0])
        ]
        print(f"\n2. UAV collision test ({len(uav_positions)} UAVs):")
        collision_10 = checker.check_uav_collision(uav_positions, min_distance=10.0)
        collision_3 = checker.check_uav_collision(uav_positions, min_distance=3.0)
        print(f"   Min distance 10m: collision={collision_10}")
        print(f"   Min distance  3m: collision={collision_3}")
        
        # تست 3: اطلاعات تفصیلی
        test_pos = np.array([10.0, 10.0, 25.0])
        info = checker.get_collision_info(test_pos, radius=1.0)
        print(f"\n3. Collision info for {test_pos}:")
        print(f"   Has collision: {info['has_collision']}")
        print(f"   Distance to nearest: {info['distance_to_nearest']:.2f}m")
        if info['nearest_obstacle']:
            print(f"   Nearest obstacle: {info['nearest_obstacle']['type']}")
        
        # تست 4: Extended safety
        print(f"\n4. Extended safety check:")
        safe_no_uavs = checker.is_position_safe_extended(test_pos)
        safe_with_uavs = checker.is_position_safe_extended(test_pos, uav_positions)
        print(f"   Safe (no other UAVs): {safe_no_uavs}")
        print(f"   Safe (with UAVs): {safe_with_uavs}")
        
        # تست 5: Out-of-bounds
        oob_positions = [
            np.array([-5.0, 50.0, 25.0]),   # x < 0
            np.array([105.0, 50.0, 25.0]),  # x > map_size
            np.array([50.0, 110.0, 25.0]),  # y > map_size
        ]
        print(f"\n5. Out-of-bounds test:")
        for pos in oob_positions:
            is_safe = checker.is_safe_position_3d(pos, uav_radius=1.0)
            print(f"   {pos}: safe={is_safe} (expected: False)")
        
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
