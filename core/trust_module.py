# core/trust_module.py
import math

class DTLCM:
    """
    Dual Trust-Level Consensus Mechanism (DTLCM)
    Hybrid analytical model combining distance and capacity metrics
    for UAV/agent initial task selection.
    Compatible with legacy SkyMind MADDPG configuration.
    """

    def __init__(self, alpha=5e-4, gamma=0.97):
        # store coefficients for hybrid decision mechanism
        self.alpha = alpha
        self.gamma = gamma

    def compute_trust(self, distance: float, capacity: float) -> float:
        """
        Compute trust metric based on legacy MADDPG–DTLCM analytical rule.
        T = exp(-alpha * distance) * (gamma * capacity)
        """
        return math.exp(-self.alpha * distance) * (self.gamma * capacity)

    def update(self, new_alpha=None, new_gamma=None):
        """Allow runtime adjustment."""
        if new_alpha is not None:
            self.alpha = new_alpha
        if new_gamma is not None:
            self.gamma = new_gamma

    def __repr__(self):
        return f"DTLCM(alpha={self.alpha}, gamma={self.gamma})"
