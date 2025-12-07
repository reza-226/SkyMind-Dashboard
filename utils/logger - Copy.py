# utils/logger.py
import os
import json
import datetime

class Logger:
    """
    Scientific execution logger for SkyMind-Dashboard.
    Automatically records informational messages, warnings, errors,
    and equilibrium metrics (U, Δ, Ω) to both console and log files.
    """

    def __init__(self, log_dir="results/logs", filename=None):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir, filename or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self._buffer = []

    def _write(self, prefix: str, message: str):
        stamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        text = f"{stamp} [{prefix}] {message}"
        print(text)
        self._buffer.append(text)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        self._buffer.clear()

    def info(self, message: str):
        """General information output."""
        self._write("INFO", message)

    def warn(self, message: str):
        """Warning message."""
        self._write("WARN", message)

    def error(self, message: str):
        """Error message."""
        self._write("ERROR", message)

    def log_equilibrium(self, U: float, delta: float, omega: float):
        """
        Log scientific equilibrium metrics extracted from environment reports.
        These appear as 'U=0.5908, Δ≈3.0%, Ω≈0.78' in `pasted-text.txt`.
        """
        msg = f"Scientific Equilibrium | U={U:.4f}, Δ={delta:.2f}%, Ω≈{omega:.2f}"
        self.info(msg)
        data = {
            "utility": U,
            "delta_percent": delta,
            "omega": omega,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        json_path = self.log_file.replace(".log", ".json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(data, jf, indent=2)
