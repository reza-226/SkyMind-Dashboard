# core/architecture_mato_uav_v2.py
import numpy as np

class DTLCMSelector:
    """Dual Trust and Load-Capacity Metric selector"""
    def __init__(self, w_dist=0.5, w_cap=0.5):
        self.wd = w_dist
        self.wc = w_cap
    def evaluate(self, uav):
        trust_score = self.wd*uav.dist + self.wc*uav.capacity
        return 1 / (1 + trust_score)

# core/architecture_mato_uav_v2.py
class MATO_UAV_v2:
    def __init__(self, trust_module=None, max_episode=2000, **kwargs):
        self.max_episode = max_episode
        self.trust_module = trust_module
        # تنظیم باقی مؤلفه‌های سیستم
        self.agents = []
        self.env = None

        if self.trust_module is not None:
            print(f"[Ninja] 🧩 Trust module injected: {self.trust_module}")
