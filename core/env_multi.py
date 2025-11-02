# =====================================================
# core/env_multi.py  (FINAL - SkyMind/UTPTR grounded)
# =====================================================
import numpy as np

class MultiUAVEnv:
    """
    Multi-UAV Environment modeled after SkyMind thesis (Malek Ashtar Univ.)
    and UTPTR stochastic game (IEEE Access 2024–25).
    """

    def __init__(self,
                 n_agents=3,
                 n_users=10,
                 dt=1.0,
                 area_size=1000.0,
                 c1=9.26e-4,
                 c2=2250.0,
                 bandwidth=1e6,
                 noise_power=1e-10,
                 alpha_delay=1.0,
                 beta_energy=1e-6,
                 gamma_eff=1e3):
        self.n_agents = n_agents
        self.n_users = n_users
        self.dt = dt
        self.area_size = area_size
        self.c1 = c1
        self.c2 = c2
        self.bandwidth = bandwidth
        self.noise_power = noise_power
        self.alpha = alpha_delay
        self.beta = beta_energy
        self.gamma = gamma_eff
        self.reset()

    # --------------------------------------------------
    def reset(self):
        """Initialize UAV and user states."""
        self.uavs = [{
            "pos": np.random.rand(2) * self.area_size,
            "velocity": np.random.uniform(5, 15),
            "theta": np.random.rand() * 2 * np.pi,
            "energy": np.random.uniform(8e4, 1.2e5),
            "E_consumed": 0.0,
            "Data_processed": 0.0
        } for _ in range(self.n_agents)]

        self.users = [{
            "pos": np.random.rand(2) * self.area_size,
            "task_size": np.random.uniform(0.5e6, 5e6)
        } for _ in range(self.n_users)]

        return self._get_state_dict()

    # --------------------------------------------------
    def _get_state_dict(self):
        uav_positions = np.array([u["pos"] for u in self.uavs])
        uav_velocities = np.array([u["velocity"] for u in self.uavs])
        uav_angles = np.array([u["theta"] for u in self.uavs])
        user_positions = np.array([u["pos"] for u in self.users])
        energies = np.array([u["energy"] for u in self.uavs])
        distances = np.array([
            np.mean([np.linalg.norm(u["pos"] - usr["pos"]) for usr in self.users])
            for u in self.uavs
        ])
        return {
            "uav_positions": uav_positions,
            "uav_velocities": uav_velocities,
            "uav_angles": uav_angles,
            "user_positions": user_positions,
            "energy": energies,
            "distances": distances
        }

    # --------------------------------------------------
    def step(self, actions):
        """Perform one simulation step for all UAVs."""
        rewards = np.zeros(self.n_agents)
        total_delay, energy_total, total_data = 0.0, 0.0, 0.0
        delays = []

        for i, (uav, act) in enumerate(zip(self.uavs, actions)):
            v = np.clip(act[0], 1.0, 30.0)
            theta = act[1]
            f = np.clip(act[2], 0.5e9, 3e9)
            o = np.clip(act[3], 0.0, 1.0)

            # --- Movement update
            uav["pos"] += v * np.array([np.cos(theta), np.sin(theta)]) * self.dt
            uav["pos"] = np.clip(uav["pos"], 0, self.area_size)
            uav["velocity"], uav["theta"] = v, theta

            # --- Nearest user
            dists = [np.linalg.norm(uav["pos"] - usr["pos"]) for usr in self.users]
            usr = self.users[int(np.argmin(dists))]
            d = max(np.min(dists), 1.0)
            h = 1.0 / (d**2)
            snr = h / self.noise_power
            R = self.bandwidth * np.log2(1 + snr)

            # --- Energy model (Propulsion + Computation + Communication)
            P_prop = self.c1 * (v ** 3) + self.c2 / v
            E_prop = P_prop * self.dt
            E_comp_local = 1e-27 * (f ** 2) * (1 - o) * usr["task_size"]
            E_comm = 1e-6 * o * usr["task_size"] / R
            E_unit = E_prop + E_comp_local + E_comm

            uav["E_consumed"] += E_unit
            uav["energy"] = max(uav["energy"] - E_unit, 0)
            energy_total += E_unit

            # --- Delay model (local + offloading)
            D_local = (1 - o) * usr["task_size"] / f
            D_off = o * (usr["task_size"] / R + usr["task_size"] / 1e9)
            D_total = D_local + D_off
            total_delay += D_total
            delays.append(D_total)

            # --- Energy efficiency metric (Eq.24 - UTPTR)
            data_proc = usr["task_size"]
            uav["Data_processed"] += data_proc
            total_data += data_proc
            E_eff = data_proc / E_unit if E_unit > 0 else 0

            # --- Reward (Eq.27a style)
            rewards[i] = -(self.alpha * D_total + self.beta * E_unit) + self.gamma * E_eff

        # --- Collect info dictionary for logging
        info = {
            "delays": delays,
            "mean_delay": np.mean(delays),
            "energy_total": energy_total,   # ← scientific + code-matched key
            "delay_total": total_delay,
            "total_data": total_data,
            "E_eff_global": (total_data / energy_total) if energy_total > 0 else 0.0
        }
        return self._get_state_dict(), rewards, False, info

    # --------------------------------------------------
    def step_multi(self, actions):
        """MADDPG-compatible multi-agent step wrapper."""
        obs_next, rewards, done, info = self.step(actions)
        next_states = [obs_next for _ in range(self.n_agents)]
        done_flags = [done] * self.n_agents
        # Return a single dict, not list, matching train_multi expectations
        return next_states, rewards, done_flags, info
