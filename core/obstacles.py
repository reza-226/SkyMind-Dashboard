"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚧 سیستم جامع مدیریت موانع
مسیر: core/obstacles.py
نویسنده: SkyMind Team
تاریخ: 1404/08/21
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import json

class ObstacleType(Enum):
    STATIC = "static"           # ساختمان، کوه
    DYNAMIC = "dynamic"         # پهپادهای دیگر، پرندگان
    NO_FLY_ZONE = "no_fly"     # مناطق ممنوعه
    WEATHER = "weather"         # شرایط جوی

class ComplexityLevel(Enum):
    SIMPLE = 1
    MEDIUM = 2
    COMPLEX = 3

@dataclass
class Obstacle:
    """کلاس پایه برای موانع"""
    id: int
    type: ObstacleType
    position: np.ndarray  # [x, y, z]
    size: Tuple[float, float, float]  # (length, width, height)
    is_active: bool = True
    penalty: float = -10.0
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type.value,
            'position': self.position.tolist(),
            'size': self.size,
            'penalty': self.penalty
        }

@dataclass
class DynamicObstacle(Obstacle):
    """مانع متحرک"""
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    trajectory: Optional[List[np.ndarray]] = None
    update_interval: float = 0.1
    
    def update_position(self, dt: float = 0.1):
        """به‌روزرسانی موقعیت بر اساس سرعت"""
        self.position += self.velocity * dt
        
        if self.trajectory is not None:
            self.trajectory.append(self.position.copy())
    
    def reverse_velocity(self):
        """تغییر جهت حرکت (bounce)"""
        self.velocity *= -1

@dataclass
class NoFlyZone(Obstacle):
    """منطقه پرواز ممنوع"""
    vertices: List[np.ndarray] = field(default_factory=list)
    altitude_range: Tuple[float, float] = (0, 100)
    bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)  # اضافه شده
    
    def is_inside(self, position: np.ndarray) -> bool:
        """
        بررسی داخل بودن نقطه در چندضلعی
        الگوریتم: Ray Casting (Point in Polygon)
        """
        x, y, z = position
        
        # بررسی ارتفاع
        if not (self.altitude_range[0] <= z <= self.altitude_range[1]):
            return False
        
        # Point-in-polygon algorithm
        n = len(self.vertices)
        inside = False
        p1x, p1y = self.vertices[0][0], self.vertices[0][1]
        
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n][0], self.vertices[i % n][1]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

class ObstacleManager:
    """مدیریت کامل موانع"""
    
    def __init__(self, 
                 complexity: ComplexityLevel, 
                 area_size: Tuple[float, float, float] = (1000, 1000, 150),
                 seed: Optional[int] = None):
        """
        Args:
            complexity: سطح پیچیدگی (SIMPLE, MEDIUM, COMPLEX)
            area_size: ابعاد محیط (x_max, y_max, z_max)
            seed: Random seed برای reproducibility
        """
        self.complexity = complexity
        self.area_size = area_size
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.obstacles: List[Obstacle] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []
        self.no_fly_zones: List[NoFlyZone] = []
        
        # آمار برخوردها
        self.collision_stats = {
            'total_collisions': 0,
            'static_collisions': 0,
            'dynamic_collisions': 0,
            'nfz_violations': 0
        }
        
        self._generate_obstacles()
        self._log_configuration()
    
    def _generate_obstacles(self):
        """تولید موانع بر اساس سطح پیچیدگی"""
        if self.complexity == ComplexityLevel.SIMPLE:
            self._generate_simple_scenario()
        elif self.complexity == ComplexityLevel.MEDIUM:
            self._generate_medium_scenario()
        else:
            self._generate_complex_scenario()
    
    def _generate_simple_scenario(self):
        """🟢 سناریوی ساده: مناطق روستایی، دشت‌های باز"""
        print("━" * 60)
        print("🟢 تولید سناریوی ساده (Rural/Open Areas)")
        print("━" * 60)
        
        # 3-5 مانع استاتیک (ساختمان‌های پراکنده)
        n_static = np.random.randint(3, 6)
        for i in range(n_static):
            pos = np.array([
                np.random.uniform(100, self.area_size[0] - 100),
                np.random.uniform(100, self.area_size[1] - 100),
                np.random.uniform(10, 40)  # ساختمان‌های کوتاه
            ])
            size = (
                np.random.uniform(20, 50),
                np.random.uniform(20, 50),
                pos[2]
            )
            
            obstacle = Obstacle(
                id=i,
                type=ObstacleType.STATIC,
                position=pos,
                size=size,
                penalty=-5.0
            )
            self.obstacles.append(obstacle)
        
        # 1-2 منطقه ممنوعه مستطیلی
        n_nfz = np.random.randint(1, 3)
        for i in range(n_nfz):
            center = np.array([
                np.random.uniform(200, self.area_size[0] - 200),
                np.random.uniform(200, self.area_size[1] - 200),
                0
            ])
            
            # مستطیل ساده
            half_w, half_h = 50, 50
            vertices = [
                center + np.array([-half_w, -half_h, 0]),
                center + np.array([half_w, -half_h, 0]),
                center + np.array([half_w, half_h, 0]),
                center + np.array([-half_w, half_h, 0])
            ]
            
            nfz = NoFlyZone(
                id=n_static + i,
                type=ObstacleType.NO_FLY_ZONE,
                position=center,
                size=(100, 100, 100),
                vertices=vertices,
                altitude_range=(0, 100),
                bounds=(center[0]-half_w, center[1]-half_h, center[0]+half_w, center[1]+half_h),
                penalty=-15.0
            )
            self.no_fly_zones.append(nfz)
        
        print(f"  ✅ {len(self.obstacles)} موانع استاتیک")
        print(f"  ✅ {len(self.no_fly_zones)} منطقه ممنوعه")
        print(f"  📊 تراکم: {self._calculate_density():.2%}")
        print("━" * 60)
    
    def _generate_medium_scenario(self):
        """🟡 سناریوی متوسط: شهرهای کوچک، نیمه‌شهری"""
        print("━" * 60)
        print("🟡 تولید سناریوی متوسط (Suburban Areas)")
        print("━" * 60)
        
        # 10-15 مانع استاتیک
        n_static = np.random.randint(10, 16)
        for i in range(n_static):
            pos = np.array([
                np.random.uniform(50, self.area_size[0] - 50),
                np.random.uniform(50, self.area_size[1] - 50),
                np.random.uniform(20, 80)
            ])
            size = (
                np.random.uniform(15, 60),
                np.random.uniform(15, 60),
                pos[2]
            )
            
            obstacle = Obstacle(
                id=i,
                type=ObstacleType.STATIC,
                position=pos,
                size=size,
                penalty=-7.0
            )
            self.obstacles.append(obstacle)
        
        # 3-5 مانع دینامیک (پهپادهای دیگر، پرندگان)
        n_dynamic = np.random.randint(3, 6)
        for i in range(n_dynamic):
            pos = np.array([
                np.random.uniform(100, self.area_size[0] - 100),
                np.random.uniform(100, self.area_size[1] - 100),
                np.random.uniform(30, 60)
            ])
            vel = np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(-1, 1)
            ])
            
            dyn_obs = DynamicObstacle(
                id=n_static + i,
                type=ObstacleType.DYNAMIC,
                position=pos,
                size=(3, 3, 2),
                velocity=vel,
                penalty=-12.0,
                trajectory=[]
            )
            self.dynamic_obstacles.append(dyn_obs)
        
        # 3-5 منطقه ممنوعه (چندضلعی)
        n_nfz = np.random.randint(3, 6)
        for i in range(n_nfz):
            n_vertices = np.random.randint(4, 7)
            center = np.array([
                np.random.uniform(150, self.area_size[0] - 150),
                np.random.uniform(150, self.area_size[1] - 150),
                0
            ])
            
            angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
            radii = np.random.uniform(40, 80, n_vertices)
            vertices = [
                center + np.array([r * np.cos(a), r * np.sin(a), 0])
                for a, r in zip(angles, radii)
            ]
            
            # محاسبه bounds
            v_array = np.array(vertices)
            bounds = (v_array[:, 0].min(), v_array[:, 1].min(), 
                     v_array[:, 0].max(), v_array[:, 1].max())
            
            nfz = NoFlyZone(
                id=n_static + n_dynamic + i,
                type=ObstacleType.NO_FLY_ZONE,
                position=center,
                size=(max(radii)*2, max(radii)*2, 120),
                vertices=vertices,
                altitude_range=(0, 120),
                bounds=bounds,
                penalty=-20.0
            )
            self.no_fly_zones.append(nfz)
        
        print(f"  ✅ {len(self.obstacles)} موانع استاتیک")
        print(f"  ✅ {len(self.dynamic_obstacles)} موانع دینامیک")
        print(f"  ✅ {len(self.no_fly_zones)} منطقه ممنوعه")
        print(f"  📊 تراکم: {self._calculate_density():.2%}")
        print("━" * 60)
    
    def _generate_complex_scenario(self):
        """🔴 سناریوی پیچیده: کلان‌شهرها، مناطق متراکم"""
        print("━" * 60)
        print("🔴 تولید سناریوی پیچیده (Dense Urban)")
        print("━" * 60)
        
        # 20-30 مانع استاتیک (آسمان‌خراش‌ها)
        n_static = np.random.randint(20, 31)
        
        # ایجاد cluster های شهری
        n_clusters = np.random.randint(3, 6)
        cluster_centers = [
            np.array([
                np.random.uniform(200, self.area_size[0] - 200),
                np.random.uniform(200, self.area_size[1] - 200)
            ])
            for _ in range(n_clusters)
        ]
        
        for i in range(n_static):
            # انتخاب cluster تصادفی
            cluster = cluster_centers[np.random.randint(0, n_clusters)]
            
            pos = np.array([
                cluster[0] + np.random.normal(0, 50),
                cluster[1] + np.random.normal(0, 50),
                np.random.uniform(30, 150)  # ساختمان‌های بلند
            ])
            
            # محدود کردن به محدوده
            pos[0] = np.clip(pos[0], 0, self.area_size[0])
            pos[1] = np.clip(pos[1], 0, self.area_size[1])
            
            size = (
                np.random.uniform(10, 80),
                np.random.uniform(10, 80),
                pos[2]
            )
            
            obstacle = Obstacle(
                id=i,
                type=ObstacleType.STATIC,
                position=pos,
                size=size,
                penalty=-10.0
            )
            self.obstacles.append(obstacle)
        
        # 10-15 مانع دینامیک (ترافیک هوایی سنگین)
        n_dynamic = np.random.randint(10, 16)
        for i in range(n_dynamic):
            pos = np.array([
                np.random.uniform(50, self.area_size[0] - 50),
                np.random.uniform(50, self.area_size[1] - 50),
                np.random.uniform(40, 100)
            ])
            vel = np.array([
                np.random.uniform(-8, 8),
                np.random.uniform(-8, 8),
                np.random.uniform(-2, 2)
            ])
            
            dyn_obs = DynamicObstacle(
                id=n_static + i,
                type=ObstacleType.DYNAMIC,
                position=pos,
                size=(4, 4, 3),
                velocity=vel,
                penalty=-15.0,
                trajectory=[]
            )
            self.dynamic_obstacles.append(dyn_obs)
        
        # 10+ منطقه ممنوعه پیچیده
        n_nfz = np.random.randint(10, 16)
        for i in range(n_nfz):
            n_vertices = np.random.randint(5, 9)
            center = np.array([
                np.random.uniform(150, self.area_size[0] - 150),
                np.random.uniform(150, self.area_size[1] - 150),
                0
            ])
            
            angles = np.sort(np.random.uniform(0, 2 * np.pi, n_vertices))
            radii = np.random.uniform(30, 100, n_vertices)
            vertices = [
                center + np.array([r * np.cos(a), r * np.sin(a), 0])
                for a, r in zip(angles, radii)
            ]
            
            # محاسبه bounds
            v_array = np.array(vertices)
            bounds = (v_array[:, 0].min(), v_array[:, 1].min(), 
                     v_array[:, 0].max(), v_array[:, 1].max())
            
            nfz = NoFlyZone(
                id=n_static + n_dynamic + i,
                type=ObstacleType.NO_FLY_ZONE,
                position=center,
                size=(max(radii)*2, max(radii)*2, 150),
                vertices=vertices,
                altitude_range=(np.random.uniform(0, 50), 150),
                bounds=bounds,
                penalty=-25.0
            )
            self.no_fly_zones.append(nfz)
        
        print(f"  ✅ {len(self.obstacles)} موانع استاتیک (clustered)")
        print(f"  ✅ {len(self.dynamic_obstacles)} موانع دینامیک")
        print(f"  ✅ {len(self.no_fly_zones)} منطقه ممنوعه")
        print(f"  📊 تراکم: {self._calculate_density():.2%}")
        print("━" * 60)
    
    def _calculate_density(self) -> float:
        """محاسبه تراکم موانع"""
        total_area = self.area_size[0] * self.area_size[1]
        occupied_area = sum(obs.size[0] * obs.size[1] for obs in self.obstacles)
        return occupied_area / total_area
    
    def _log_configuration(self):
        """ذخیره تنظیمات برای reproducibility"""
        import os
        os.makedirs('results', exist_ok=True)
        
        config = {
            'complexity': self.complexity.name,
            'area_size': self.area_size,
            'seed': self.seed,
            'n_static': len(self.obstacles),
            'n_dynamic': len(self.dynamic_obstacles),
            'n_nfz': len(self.no_fly_zones),
            'density': self._calculate_density()
        }
        
        with open(f'results/obstacle_config_{self.complexity.name}.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def check_collision(self, 
                       position: np.ndarray, 
                       safety_margin: float = 5.0) -> Tuple[bool, float, Dict[str, int]]:
        """
        بررسی برخورد با موانع
        
        Args:
            position: موقعیت UAV [x, y, z]
            safety_margin: حاشیه امنیتی (متر)
        
        Returns:
            (has_collision, total_penalty, collision_details)
        """
        penalty = 0.0
        details = {
            'static': 0,
            'dynamic': 0,
            'nfz': 0
        }
        
        # بررسی موانع استاتیک
        for obs in self.obstacles:
            if not obs.is_active:
                continue
            
            # محاسبه فاصله
            dist = np.linalg.norm(position[:2] - obs.position[:2])
            
            # بررسی برخورد
            if (dist < (max(obs.size[:2]) / 2 + safety_margin) and 
                position[2] <= obs.size[2]):
                penalty += obs.penalty
                details['static'] += 1
                self.collision_stats['static_collisions'] += 1
        
        # بررسی موانع دینامیک
        for dyn_obs in self.dynamic_obstacles:
            if not dyn_obs.is_active:
                continue
            
            dist = np.linalg.norm(position - dyn_obs.position)
            if dist < (max(dyn_obs.size) / 2 + safety_margin):
                penalty += dyn_obs.penalty
                details['dynamic'] += 1
                self.collision_stats['dynamic_collisions'] += 1
        
        # بررسی مناطق ممنوعه
        for nfz in self.no_fly_zones:
            if not nfz.is_active:
                continue
            
            if nfz.is_inside(position):
                penalty += nfz.penalty
                details['nfz'] += 1
                self.collision_stats['nfz_violations'] += 1
        
        has_collision = penalty < 0
        if has_collision:
            self.collision_stats['total_collisions'] += 1
        
        return (has_collision, penalty, details)
    
    def update(self, dt: float = 0.1):
        """به‌روزرسانی موانع دینامیک"""
        for dyn_obs in self.dynamic_obstacles:
            # ذخیره موقعیت قبلی
            old_pos = dyn_obs.position.copy()
            
            # به‌روزرسانی موقعیت
            dyn_obs.update_position(dt)
            
            # بررسی خروج از محدوده و bounce
            if not (0 <= dyn_obs.position[0] <= self.area_size[0]):
                dyn_obs.position = old_pos
                dyn_obs.velocity[0] *= -1
            
            if not (0 <= dyn_obs.position[1] <= self.area_size[1]):
                dyn_obs.position = old_pos
                dyn_obs.velocity[1] *= -1
            
            if not (10 <= dyn_obs.position[2] <= self.area_size[2]):
                dyn_obs.position = old_pos
                dyn_obs.velocity[2] *= -1
    
    def get_nearest_obstacles(self, 
                             position: np.ndarray, 
                             k: int = 5) -> List[Tuple[Obstacle, float]]:
        """
        یافتن k نزدیک‌ترین مانع
        
        Returns:
            لیستی از (obstacle, distance)
        """
        all_obs = self.obstacles + self.dynamic_obstacles
        
        distances = [
            (obs, np.linalg.norm(position - obs.position))
            for obs in all_obs if obs.is_active
        ]
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def get_state_vector(self, position: np.ndarray, k: int = 3) -> np.ndarray:
        """
        تبدیل موانع به feature vector برای RL
        
        Args:
            position: موقعیت فعلی UAV
            k: تعداد نزدیک‌ترین موانع
        
        Returns:
            Vector با shape (k * 4,) شامل [dx, dy, dz, type]
        """
        nearest = self.get_nearest_obstacles(position, k)
        
        features = []
        for obs, dist in nearest:
            relative_pos = obs.position - position
            obs_type = 1.0 if isinstance(obs, DynamicObstacle) else 0.0
            features.extend([*relative_pos, obs_type])
        
        # Padding اگر کمتر از k مانع وجود داشت
        while len(features) < k * 4:
            features.extend([0, 0, 0, 0])
        
        return np.array(features[:k * 4])
    
    def get_collision_stats(self) -> Dict:
        """دریافت آمار برخوردها"""
        return self.collision_stats.copy()
    
    def reset_stats(self):
        """ریست آمار"""
        self.collision_stats = {
            'total_collisions': 0,
            'static_collisions': 0,
            'dynamic_collisions': 0,
            'nfz_violations': 0
        }
    
    def visualize(self, ax, show_safety_margin: bool = False):
        """
        رسم موانع در matplotlib 3D
        
        Args:
            ax: Axes3D object
            show_safety_margin: نمایش حاشیه امنیتی
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # موانع استاتیک (مکعب)
        for obs in self.obstacles:
            if not obs.is_active:
                continue
            
            x, y, z = obs.position
            dx, dy, dz = obs.size
            
            vertices = [
                [x-dx/2, y-dy/2, 0],
                [x+dx/2, y-dy/2, 0],
                [x+dx/2, y+dy/2, 0],
                [x-dx/2, y+dy/2, 0],
                [x-dx/2, y-dy/2, dz],
                [x+dx/2, y-dy/2, dz],
                [x+dx/2, y+dy/2, dz],
                [x-dx/2, y+dy/2, dz]
            ]
            
            faces = [
                [vertices[j] for j in [0, 1, 5, 4]],
                [vertices[j] for j in [7, 6, 2, 3]],
                [vertices[j] for j in [0, 3, 7, 4]],
                [vertices[j] for j in [1, 2, 6, 5]],
                [vertices[j] for j in [0, 1, 2, 3]],
                [vertices[j] for j in [4, 5, 6, 7]]
            ]
            
            ax.add_collection3d(Poly3DCollection(
                faces, facecolors='darkgray', linewidths=0.5,
                edgecolors='black', alpha=0.4
            ))
        
        # موانع دینامیک (کره)
        for dyn_obs in self.dynamic_obstacles:
            if not dyn_obs.is_active:
                continue
            
            x, y, z = dyn_obs.position
            r = max(dyn_obs.size) / 2
            
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            xs = x + r * np.outer(np.cos(u), np.sin(v))
            ys = y + r * np.outer(np.sin(u), np.sin(v))
            zs = z + r * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(xs, ys, zs, color='red', alpha=0.5, linewidth=0)
            
            # رسم بردار سرعت
            ax.quiver(x, y, z, 
                     dyn_obs.velocity[0], dyn_obs.velocity[1], dyn_obs.velocity[2],
                     color='orange', arrow_length_ratio=0.3, linewidth=2)
        
        # مناطق ممنوعه
        for nfz in self.no_fly_zones:
            if not nfz.is_active:
                continue
            
            vertices = np.array(nfz.vertices)
            xs = np.append(vertices[:, 0], vertices[0, 0])
            ys = np.append(vertices[:, 1], vertices[0, 1])
            
            # کف و سقف
            ax.plot(xs, ys, nfz.altitude_range[0], 
                   'r--', linewidth=2, alpha=0.7, label='No-Fly Zone')
            ax.plot(xs, ys, nfz.altitude_range[1], 
                   'r--', linewidth=2, alpha=0.7)
            
            # پر کردن کف - اصلاح شده ✅
            verts_bottom = [list(zip(xs, ys, [nfz.altitude_range[0]] * len(xs)))]
            ax.add_collection3d(Poly3DCollection(
                verts_bottom, facecolors='red', alpha=0.1
            ))
        
        # تنظیمات محور
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_xlim([0, self.area_size[0]])
        ax.set_ylim([0, self.area_size[1]])
        ax.set_zlim([0, self.area_size[2]])


# ═══════════════════════════════════════════════════════════════════
# 🧪 تست سریع
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 تست ObstacleManager\n")
    
    for level in ComplexityLevel:
        manager = ObstacleManager(complexity=level, seed=42)
        
        # تست برخورد
        test_pos