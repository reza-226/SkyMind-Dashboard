"""
ObstacleManager Patch for 3D Map Size Support
==============================================
این پچ ObstacleManager را برای پذیرش map_size به صورت [x, y, z] اصلاح می‌کند.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Union, List, Tuple

# اضافه کردن root به path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def patch_obstacle_manager():
    """
    اعمال پچ به ObstacleManager برای پشتیبانی از map_size سه‌بعدی
    
    Returns:
        tuple: (patched_methods, errors)
    """
    # Import هوشمند
    try:
        from core.obstacles import ObstacleManager
    except ImportError:
        try:
            from obstacles import ObstacleManager
        except ImportError as e:
            return [], [f"Cannot import ObstacleManager: {e}"]
    
    patched_methods = []
    errors = []
    
    # ذخیره متد اصلی __init__
    original_init = ObstacleManager.__init__
    
    def patched_init(self, map_size: Union[float, List[float]], num_obstacles: int = 20, 
                     obstacle_radius: float = 5.0, seed: int = None):
        """
        مقداردهی اولیه با پشتیبانی از map_size سه‌بعدی
        
        Args:
            map_size: اندازه نقشه - می‌تواند باشد:
                     - عدد (scalar): برای محیط مربعی 2D
                     - [width, height]: برای محیط مستطیلی 2D  
                     - [width, height, altitude]: برای محیط 3D (فقط x,y برای موانع استفاده می‌شود)
            num_obstacles: تعداد موانع
            obstacle_radius: شعاع موانع
            seed: seed برای تولید تصادفی
        """
        # تبدیل map_size به فرمت 2D برای موانع
        if isinstance(map_size, (list, tuple, np.ndarray)):
            if len(map_size) >= 2:
                # استفاده از x, y برای موانع 2D (نادیده گرفتن z)
                self.map_size_2d = np.array(map_size[:2], dtype=float)
                self.map_size_full = np.array(map_size, dtype=float)
            else:
                # اگر فقط یک عدد داریم، تبدیل به مربع
                size = float(map_size[0])
                self.map_size_2d = np.array([size, size], dtype=float)
                self.map_size_full = self.map_size_2d
        else:
            # scalar: تبدیل به مربع 2D
            size = float(map_size)
            self.map_size_2d = np.array([size, size], dtype=float)
            self.map_size_full = self.map_size_2d
        
        # ذخیره map_size اصلی برای سازگاری با کد قدیمی
        self.map_size = self.map_size_2d[0] if np.allclose(self.map_size_2d[0], self.map_size_2d[1]) else self.map_size_2d
        
        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        self.rng = np.random.default_rng(seed)
        self.obstacles = []
        
        # تولید موانع
        self._generate_obstacles()
    
    # ذخیره متد اصلی _generate_obstacles
    original_generate = ObstacleManager._generate_obstacles
    
    def patched_generate_obstacles(self):
        """تولید موانع تصادفی با استفاده از map_size_2d"""
        self.obstacles = []
        margin = 10.0
        
        for _ in range(self.num_obstacles):
            # محاسبه حدود برای هر بعد
            x_min = margin
            x_max = max(self.map_size_2d[0] - margin, margin + 1)
            y_min = margin
            y_max = max(self.map_size_2d[1] - margin, margin + 1)
            
            # تولید موقعیت تصادفی 2D
            x = self.rng.uniform(x_min, x_max)
            y = self.rng.uniform(y_min, y_max)
            position = np.array([x, y], dtype=float)
            
            self.obstacles.append({
                'position': position,
                'radius': self.obstacle_radius,
                'type': 'cylinder'  # موانع استوانه‌ای 2D
            })
    
    # اعمال پچ‌ها
    try:
        ObstacleManager.__init__ = patched_init
        patched_methods.append('__init__')
    except Exception as e:
        errors.append(f"Failed to patch __init__: {e}")
    
    try:
        ObstacleManager._generate_obstacles = patched_generate_obstacles
        patched_methods.append('_generate_obstacles')
    except Exception as e:
        errors.append(f"Failed to patch _generate_obstacles: {e}")
    
    return patched_methods, errors


def test_patch():
    """تست پچ ObstacleManager"""
    print("=" * 80)
    print("Testing ObstacleManager Patch")
    print("=" * 80)
    print(f"📂 Running from: {Path(__file__).absolute()}")
    print(f"📂 Project root: {project_root}")
    print()
    
    patched_methods, errors = patch_obstacle_manager()
    
    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  ✗ {error}")
        return False
    
    print("✅ Patched methods:")
    for method in patched_methods:
        print(f"  ✓ {method}")
    
    print("\n" + "-" * 80)
    print("Testing with different map_size formats...")
    print("-" * 80)
    
    # Import هوشمند
    try:
        from core.obstacles import ObstacleManager
    except ImportError:
        from obstacles import ObstacleManager
    
    test_cases = [
        ("Scalar (100.0)", 100.0),
        ("2D list ([100.0, 100.0])", [100.0, 100.0]),
        ("3D list ([100.0, 100.0, 50.0])", [100.0, 100.0, 50.0]),
        ("Numpy array 2D", np.array([100.0, 100.0])),
        ("Numpy array 3D", np.array([100.0, 100.0, 50.0])),
    ]
    
    for test_name, map_size in test_cases:
        try:
            print(f"\n📌 Test: {test_name}")
            print(f"   Input: {map_size}")
            
            om = ObstacleManager(map_size=map_size, num_obstacles=5, seed=42)
            
            print(f"   ✓ Created ObstacleManager")
            print(f"   - map_size (original): {om.map_size}")
            print(f"   - map_size_2d: {om.map_size_2d}")
            print(f"   - map_size_full: {om.map_size_full}")
            print(f"   - Number of obstacles: {len(om.obstacles)}")
            
            if len(om.obstacles) > 0:
                print(f"   - Sample obstacle pos: {om.obstacles[0]['position']}")
                print(f"   - Sample obstacle radius: {om.obstacles[0]['radius']}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    
    return True


if __name__ == '__main__':
    success = test_patch()
    exit(0 if success else 1)
