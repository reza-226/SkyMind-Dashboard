"""
generate_comparison_tables.py
=============================
Step 1.2 – Generate LaTeX comparison tables based on realtime_cache.pkl.
Handles structure where 'episodes' is an int and histories are lists.
"""

import pickle
import numpy as np
from pathlib import Path
import os


def generate_latex_table(data_dict, caption, label, cols, filename):
    """Generate LaTeX table from dictionary and save."""
    header = " & ".join(cols) + " \\\\ \\hline\n"
    body = ""
    for k, v in data_dict.items():
        if isinstance(v, (list, np.ndarray)):
            v = np.mean(v)
        body += f"{k} & {v:.4f} \\\\ \n"
    table_code = (
        f"\\begin{{table}}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{|l|c|}}\n\\hline\n{header}{body}\\hline\n"
        f"\\end{{tabular}}\n\\end{{table}}\n"
    )
    with open(filename, "w", encoding="utf-8") as f:
        f.write(table_code)
    print(f"✅ Table saved: {filename}")


def main():
    project_root = Path(__file__).parent.parent.parent.parent
    cache_path = project_root / "analysis" / "realtime" / "realtime_cache.pkl"
    tables_dir = project_root / "analysis" / "realtime" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    # Ensure valid numeric data
    mean_energy = float(cache.get("mean_Energy_J", 0))
    mean_delay = float(cache.get("mean_Delay_ms", 0))
    mean_u = float(cache.get("mean_U", 0))
    mean_energy_red = float(cache.get("mean_Energy_Reduction_%", 0))
    mean_delay_red = float(cache.get("mean_Delay_Reduction_%", 0))
    episodes = int(cache.get("episodes", 0))
    duration = float(cache.get("duration_sec", 0))

    # Derived metrics
    improvement_index = (mean_energy_red + mean_delay_red) / 2
    stability_index = np.mean(cache["Omega_history"]) if "Omega_history" in cache else 0
    efficiency_ratio = mean_u / (mean_delay * mean_energy / 1000 + 1e-9)

    # --- Table 1: System Performance
    performance = {
        "Mean Utility": mean_u,
        "Mean Energy (J)": mean_energy,
        "Mean Delay (ms)": mean_delay,
        "Efficiency Ratio": efficiency_ratio,
    }
    generate_latex_table(performance,
                         "System Performance Summary",
                         "tab:performance_summary",
                         ["Metric", "Value"],
                         tables_dir / "table_performance.tex")

    # --- Table 2: Improvements over Baselines
    improvements = {
        "Energy Reduction (%)": mean_energy_red,
        "Delay Reduction (%)": mean_delay_red,
        "Overall Improvement Index": improvement_index,
    }
    generate_latex_table(improvements,
                         "Improvement Over Baselines",
                         "tab:improvement",
                         ["Metric", "Value"],
                         tables_dir / "table_improvement.tex")

    # --- Table 3: Convergence and Stability
    convergence = {
        "Episodes": episodes,
        "Training Duration (sec)": duration,
        "Stability Index (Ω)": stability_index,
        "Final Utility": mean_u,
    }
    generate_latex_table(convergence,
                         "Convergence and Stability Statistics",
                         "tab:convergence",
                         ["Parameter", "Value"],
                         tables_dir / "table_convergence.tex")

    # --- Table 4: Statistical Summary
    histories = ["U_history", "Energy_history", "Delay_history"]
    stats = {}
    for h in histories:
        if h in cache:
            arr = np.array(cache[h])
            stats[h.replace("_history", "").capitalize() + " Mean"] = np.mean(arr)
            stats[h.replace("_history", "").capitalize() + " Std"] = np.std(arr)
    generate_latex_table(stats,
                         "Statistical Summary of Key Metrics",
                         "tab:statistics",
                         ["Metric", "Value"],
                         tables_dir / "table_statistics.tex")

    print("\n🎉 All comparison tables generated successfully!")


if __name__ == "__main__":
    main()
