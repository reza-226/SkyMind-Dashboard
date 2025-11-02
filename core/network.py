# core/network.py
import numpy as np

class ChannelModel:
    def __init__(self, frequency_ghz=2.4, shadowing_db=2.0):
        self.freq = frequency_ghz
        self.shadowing = shadowing_db

    def path_loss_fspl(self, distance_m):
        """Free‑Space Path Loss in dB"""
        c = 3e8
        lam = c / (self.freq * 1e9)
        return 20 * np.log10(4 * np.pi * distance_m / lam)

    def snr(self, tx_power_dbm, distance_m, noise_dbm=-90):
        """Compute SNR (Signal‑to‑Noise Ratio)"""
        pl = self.path_loss_fspl(distance_m) + np.random.randn() * self.shadowing
        return tx_power_dbm - pl - noise_dbm
