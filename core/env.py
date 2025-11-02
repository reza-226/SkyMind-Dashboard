# ==========================================
# core/env.py
# SkyMind Environment (Phase 3 - DRL Setup)
# Grounded in Payannameh (Ch.2-6) + ECORI Paper
# ==========================================

import numpy as np
import random

class SkyMindEnv:
    """
    Environment model for UAV-assisted MEC system 
    based on the SkyMind framework defined in the thesis.
    Integrates channel quality (SNR), queue utilization (ρ),
    and reputation (from ECORI mechanism).
    """

    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2, k=1.2, seed=42):
        np.random.seed(seed)
        random.seed(seed)

        # Weighted coefficients from the multi-objective function (Chapter 6)
        self.alpha = alpha  # weight for delay
        self.beta = beta    # weight for energy
        self.gamma = gamma  # weight for reputation
        self.k = k          # exponential coefficient for reputation from ECORI

        # Action space: 0-local, 1-offload to UAV, 2-offload to Edge
        self.actions = [0, 1, 2]

        # Observation/state placeholders
        self.state = None
        self.reset()

    def reset(self):
        """ Initialize environment with typical scenario values """
        # Typical values from validated modules & thesis
        self.SNR = np.clip(np.random.normal(11.3, 2.0), 5, 18)       # dB
        self.queue_rho = np.clip(np.random.uniform(0.45, 0.80), 0, 1)
        self.task_size = np.clip(np.random.uniform(1, 10), 0.1, 20)  # MB
        self.energy_base = np.random.uniform(0.2, 0.5)               # baseline energy coefficient
        self.delay_base = 0.722                                       # from queue model chapter 6
        self.state = self._compose_state()
        return self.state

    def _compose_state(self):
        """ Compose state vector based on real parameters """
        reputation = np.exp(-self.k * self.queue_rho)
        return np.array([self.SNR, self.queue_rho, reputation, self.task_size], dtype=float)

    def step(self, action):
        """
        Perform one simulation step based on action type.
        Returns: next_state, reward, done, info
        """

        # Simulate channel variation and task load dynamics (per thesis section 2–6)
        snr_variation = np.random.normal(0, 0.8)
        load_variation = np.random.normal(0, 0.05)

        self.SNR = np.clip(self.SNR + snr_variation, 5, 20)
        self.queue_rho = np.clip(self.queue_rho + load_variation, 0.1, 0.95)

        # Compute updated reputation
        reputation = np.exp(-self.k * self.queue_rho)

        # Delay & Energy models
        # Base delay influenced by rho (congestion) and inverse of SNR (transmission rate)
        delay = self.delay_base * (1 + 1.2*self.queue_rho) * (1 + (10/self.SNR))
        # Base energy influenced by task size and communication mode
        if action == 0:      # local processing
            energy = self.energy_base * 0.8
        elif action == 1:    # offload to UAV
            energy = self.energy_base * (1 + 0.5*(10/self.SNR))
        elif action == 2:    # offload to Edge
            energy = self.energy_base * (1.2 + 0.2*(10/self.SNR))
        else:
            raise ValueError("Invalid action")

        # Reward function derived from chapter 6 eq.(J)
        reward = - (self.alpha*delay + self.beta*energy) - self.gamma*(1 - reputation)

        # Update next state
        self.state = self._compose_state()
        done = False

        info = {
            "delay_s": round(delay, 4),
            "energy_J": round(energy, 4),
            "reputation": round(reputation, 4),
            "snr_db": round(self.SNR, 2),
        }
        return self.state, reward, done, info

    def action_space_sample(self):
        return random.choice(self.actions)

    def observation_space_dim(self):
        return len(self.state)

# ==========================================
# Example standalone test
# ==========================================
if __name__ == "__main__":
    env = SkyMindEnv()
    state = env.reset()
    print("Initial state:", state)

    for t in range(3):
        act = env.action_space_sample()
        next_state, reward, done, info = env.step(act)
        print(f"Step {t+1} | Action={act}, Reward={reward:.4f}, Info={info}")
