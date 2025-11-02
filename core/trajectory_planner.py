# core/trajectory_planner.py
import numpy as np

class TrajectoryPlanner:
    """
    مدل حرکت و مصرف انرژی برای پهپادها بر مبنای مدل SkyMind و UTPTR.
    v_i : سرعت پهپاد (m/s)
    theta_i : زاویه حرکت (رادیان)
    """

    def __init__(self, v_max=25.0, delta_t=1.0, c1=9.26e-4, c2=2250.0):
        self.v_max = v_max        # سرعت بیشینه
        self.dt = delta_t         # گام زمانی (s)
        self.c1, self.c2 = c1, c2 # ضرایب انرژی طرح‌شده در Eq.29

    def step(self, pos, velocity, theta):
        v = np.clip(velocity, 0.1, self.v_max)
        dx = v * np.cos(theta) * self.dt
        dy = v * np.sin(theta) * self.dt
        new_pos = pos + np.array([dx, dy])

        # انرژی پروازی مطابق Eq.(29): P=c1*v^3 + c2/v
        power_prop = self.c1 * (v ** 3) + self.c2 / v
        E_prop = power_prop * self.dt
        return new_pos, E_prop
