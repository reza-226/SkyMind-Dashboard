# core/env_multi_patch.py
"""
پچ برای رفع مشکل NumPy Random API در env_multi.py
"""

from core.env_multi import MultiUAVEnv
import numpy as np


def patch_env_multi():
    """اعمال پچ برای سازگاری با NumPy جدید"""
    
    # ذخیره متد اصلی
    original_get_observations = MultiUAVEnv._get_observations
    
    def patched_get_observations(self):
        """نسخه پچ‌شده _get_observations با API صحیح NumPy"""
        observations = {}
        
        for i in range(self.num_uavs):
            uav_id = f'agent_{i}'
            uav = self.uavs[i]
            
            # موقعیت UAV
            position = uav['position']
            velocity = uav['velocity']
            
            # وضعیت محاسباتی
            compute_load = uav['compute_load']
            energy = uav['energy']
            
            # وضعیت کاربران
            user_states = []
            for user in self.ground_users:
                distance = np.linalg.norm(position[:2] - user['position'])
                # استفاده از random() به جای rand()
                has_task = 1.0 if self.np_random.random() < self.task_arrival_rate else 0.0
                task_size = user.get('task_size', 0.0)
                
                user_states.extend([
                    distance / 1000.0,  # نرمال‌سازی
                    has_task,
                    task_size / 1000.0
                ])
            
            # ترکیب observation
            obs = np.concatenate([
                position / [1000, 1000, 200],  # نرمال‌سازی موقعیت
                velocity / 10.0,  # نرمال‌سازی سرعت
                [compute_load / 100.0],  # نرمال‌سازی بار محاسباتی
                [energy / 100.0],  # نرمال‌سازی انرژی
                user_states
            ])
            
            observations[uav_id] = obs.astype(np.float32)
        
        return observations
    
    # اعمال پچ
    MultiUAVEnv._get_observations = patched_get_observations
    print("✅ env_multi.py patched: Fixed np_random.rand() -> np_random.random()")


if __name__ == "__main__":
    patch_env_multi()
    print("Patch applied successfully!")
