# ===============================================================
#  analysis/realtime/report_ch5_generator.py (v6.2 - Fixed)
#  تولید نمودارها و جداول برای فصل 5 - با Energy/Delay کامل
# ===============================================================

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# تنظیمات فونت فارسی
plt.rcParams['font.family'] = ['Vazirmatn', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CACHE_FILE = BASE_DIR / "realtime_cache.pkl"
PARETO_FILE = BASE_DIR / "pareto_snapshot.json"

print("=" * 70)
print("📊 شروع تولید گزارش‌های فصل 5 (v6.2 - با Energy/Delay)")
print("=" * 70)

# ===============================================================
# بارگذاری داده‌ها
# ===============================================================
print("\n[1/11] بارگذاری داده‌ها...")

with open(CACHE_FILE, 'rb') as f:
    cache = pickle.load(f)

with open(PARETO_FILE, 'r') as f:
    pareto = json.load(f)

solutions = pareto['solutions']
N = len(solutions)

# استخراج معیارها
episodes = list(range(1, N + 1))
U_values = [s['U'] for s in solutions]
Delta_values = [s['Δ'] for s in solutions]
Omega_values = [s['Ω'] for s in solutions]
Energy_values = [s['Energy_J'] for s in solutions]
Delay_values = [s['Delay_ms'] for s in solutions]

print(f"✅ {N} راه‌حل بارگذاری شد (U, Δ, Ω, Energy, Delay)")

# ===============================================================
# نمودار 1: Utility Convergence
# ===============================================================
print("\n[2/11] تولید نمودار همگرایی Utility...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(episodes, U_values, color='#2E86AB', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode', fontsize=14, weight='bold')
ax.set_ylabel('Utility (U)', fontsize=14, weight='bold')
ax.set_title('Utility Function Convergence', fontsize=16, weight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, N)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_utility_convergence.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig1_utility_convergence.pdf", bbox_inches='tight')
plt.close()

print(f"✅ ذخیره: {FIGURES_DIR / 'fig1_utility_convergence.png'}")

# ===============================================================
# نمودار 2: Error Rate
# ===============================================================
print("\n[3/11] تولید نمودار Error Rate...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(episodes, Delta_values, color='#A23B72', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode', fontsize=14, weight='bold')
ax.set_ylabel('Error Rate (%)', fontsize=14, weight='bold')
ax.set_title('Classification Error Rate (Δ)', fontsize=16, weight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, N)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_error_rate.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig2_error_rate.pdf", bbox_inches='tight')
plt.close()

print(f"✅ ذخیره: {FIGURES_DIR / 'fig2_error_rate.png'}")

# ===============================================================
# نمودار 3: Stability
# ===============================================================
print("\n[4/11] تولید نمودار Stability...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(episodes, Omega_values, color='#F18F01', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode', fontsize=14, weight='bold')
ax.set_ylabel('Stability (Ω)', fontsize=14, weight='bold')
ax.set_title('System Stability Metric (Ω)', fontsize=16, weight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, N)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_stability.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig3_stability.pdf", bbox_inches='tight')
plt.close()

print(f"✅ ذخیره: {FIGURES_DIR / 'fig3_stability.png'}")

# ===============================================================
# نمودار 4: Pareto Front (2D)
# ===============================================================
print("\n[5/11] تولید نمودار Pareto Front...")

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(Delta_values, U_values, 
                    c=Omega_values, cmap='viridis', 
                    s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Stability (Ω)', fontsize=12, weight='bold')

ax.set_xlabel('Error Rate (Δ) [%]', fontsize=14, weight='bold')
ax.set_ylabel('Utility (U)', fontsize=14, weight='bold')
ax.set_title('Pareto Front: Utility vs Error Rate', fontsize=16, weight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_pareto_front.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig4_pareto_front.pdf", bbox_inches='tight')
plt.close()

print(f"✅ ذخیره: {FIGURES_DIR / 'fig4_pareto_front.png'}")

# ===============================================================
# نمودار 5: 3D Objective Space
# ===============================================================
print("\n[6/11] تولید نمودار 3D...")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(U_values, Delta_values, Omega_values,
                    c=episodes, cmap='plasma', s=30, alpha=0.6)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
cbar.set_label('Episode', fontsize=12, weight='bold')

ax.set_xlabel('Utility (U)', fontsize=12, weight='bold')
ax.set_ylabel('Error Rate (Δ) [%]', fontsize=12, weight='bold')
ax.set_zlabel('Stability (Ω)', fontsize=12, weight='bold')
ax.set_title('3D Objective Space (U, Δ, Ω)', fontsize=14, weight='bold', pad=20)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig5_3d_space.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig5_3d_space.pdf", bbox_inches='tight')
plt.close()

print(f"✅ ذخیره: {FIGURES_DIR / 'fig5_3d_space.png'}")

# ===============================================================
# نمودار 6: Energy Consumption (جدید)
# ===============================================================
print("\n[7/11] تولید نمودار Energy Consumption...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(episodes, Energy_values, color='#06A77D', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode', fontsize=14, weight='bold')
ax.set_ylabel('Energy Consumption (J)', fontsize=14, weight='bold')
ax.set_title('Energy Consumption over Episodes', fontsize=16, weight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, N)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig6_energy_consumption.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig6_energy_consumption.pdf", bbox_inches='tight')
plt.close()

print(f"✅ ذخیره: {FIGURES_DIR / 'fig6_energy_consumption.png'}")

# ===============================================================
# نمودار 7: Delay Performance (جدید)
# ===============================================================
print("\n[8/11] تولید نمودار Delay Performance...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(episodes, Delay_values, color='#D90368', linewidth=2, alpha=0.8)
ax.set_xlabel('Episode', fontsize=14, weight='bold')
ax.set_ylabel('Processing Delay (ms)', fontsize=14, weight='bold')
ax.set_title('Processing Delay over Episodes', fontsize=16, weight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, N)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig7_delay_performance.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig7_delay_performance.pdf", bbox_inches='tight')
plt.close()

print(f"✅ ذخیره: {FIGURES_DIR / 'fig7_delay_performance.png'}")

# ===============================================================
# نمودار 8: Energy-Delay Trade-off (جدید)
# ===============================================================
print("\n[9/11] تولید نمودار Energy-Delay Trade-off...")

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(Energy_values, Delay_values, 
                    c=U_values, cmap='coolwarm', 
                    s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Utility (U)', fontsize=12, weight='bold')

ax.set_xlabel('Energy Consumption (J)', fontsize=14, weight='bold')
ax.set_ylabel('Processing Delay (ms)', fontsize=14, weight='bold')
ax.set_title('Energy-Delay Trade-off', fontsize=16, weight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig8_energy_delay_tradeoff.png", dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / "fig8_energy_delay_tradeoff.pdf", bbox_inches='tight')
plt.close()

print(f"✅ ذخیره: {FIGURES_DIR / 'fig8_energy_delay_tradeoff.png'}")

# ===============================================================
# جدول LaTeX (نسخه کامل با Energy/Delay)
# ===============================================================
print("\n[10/11] تولید جدول LaTeX...")

# محاسبه میانگین‌ها از cache
mean_energy = cache.get('mean_Energy_J', np.mean(Energy_values))
mean_delay = cache.get('mean_Delay_ms', np.mean(Delay_values))

# فرمت کردن رشته LaTeX با استفاده از raw strings
latex_table = r"""\begin{table}[htbp]
\centering
\caption{Performance Metrics: Initial vs Final Solutions}
\label{tab:results}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Metric} & \textbf{Initial} & \textbf{Final} & \textbf{Improvement} \\ \hline
Utility (U) & """ + f"{U_values[0]:.4f}" + r""" & """ + f"{U_values[-1]:.4f}" + r""" & """ + f"{((U_values[-1]-U_values[0])/U_values[0]*100):+.2f}" + r"""\% \\ \hline
Error Rate ($\Delta$) & """ + f"{Delta_values[0]:.2f}" + r"""\% & """ + f"{Delta_values[-1]:.2f}" + r"""\% & """ + f"{((Delta_values[0]-Delta_values[-1])/Delta_values[0]*100):+.2f}" + r"""\% \\ \hline
Stability ($\Omega$) & """ + f"{Omega_values[0]:.2f}" + r""" & """ + f"{Omega_values[-1]:.2f}" + r""" & """ + f"{((Omega_values[-1]-Omega_values[0])/Omega_values[0]*100):+.2f}" + r"""\% \\ \hline
Energy (J) & """ + f"{Energy_values[0]:.4f}" + r""" & """ + f"{Energy_values[-1]:.4f}" + r""" & """ + f"{((Energy_values[0]-Energy_values[-1])/Energy_values[0]*100):+.2f}" + r"""\% \\ \hline
Delay (ms) & """ + f"{Delay_values[0]:.2f}" + r""" & """ + f"{Delay_values[-1]:.2f}" + r""" & """ + f"{((Delay_values[0]-Delay_values[-1])/Delay_values[0]*100):+.2f}" + r"""\% \\ \hline
\textbf{Mean Values} & \multicolumn{3}{c|}{$U=$""" + f"{cache['mean_U']:.4f}" + r""", $\Delta=$""" + f"{cache['mean_Delta']:.2f}" + r"""\%, $\Omega=$""" + f"{cache['mean_Omega']:.2f}" + r""", $E=$""" + f"{mean_energy:.4f}" + r"""J, $D=$""" + f"{mean_delay:.2f}" + r"""ms} \\ \hline
\end{tabular}
\end{table}"""

# ذخیره جدول
with open(FIGURES_DIR / "table_results.tex", 'w', encoding='utf-8') as f:
    f.write(latex_table)

print(f"✅ ذخیره: {FIGURES_DIR / 'table_results.tex'}")

# ===============================================================
# جدول مقایسه مختصر (برای TikZ)
# ===============================================================
print("\n[11/11] تولید جدول مختصر...")

short_table = r"""\begin{tabular}{|l|c|c|}
\hline
\textbf{Metric} & \textbf{Initial} & \textbf{Final} \\ \hline
Utility (U) & """ + f"{U_values[0]:.4f}" + r""" & """ + f"{U_values[-1]:.4f}" + r""" \\ \hline
Error Rate ($\Delta$) & """ + f"{Delta_values[0]:.2f}" + r"""\% & """ + f"{Delta_values[-1]:.2f}" + r"""\% \\ \hline
Stability ($\Omega$) & """ + f"{Omega_values[0]:.2f}" + r""" & """ + f"{Omega_values[-1]:.2f}" + r""" \\ \hline
Energy (J) & """ + f"{Energy_values[0]:.4f}" + r""" & """ + f"{Energy_values[-1]:.4f}" + r""" \\ \hline
Delay (ms) & """ + f"{Delay_values[0]:.2f}" + r""" & """ + f"{Delay_values[-1]:.2f}" + r""" \\ \hline
\end{tabular}"""

with open(FIGURES_DIR / "table_short.tex", 'w', encoding='utf-8') as f:
    f.write(short_table)

print(f"✅ ذخیره: {FIGURES_DIR / 'table_short.tex'}")

# ===============================================================
# خلاصه نهایی
# ===============================================================
print("\n" + "="*70)
print("✅ تولید گزارش‌ها کامل شد!")
print("="*70)

print(f"\n📁 فایل‌های تولید شده در: {FIGURES_DIR}")

print("\n📊 نمودارهای تولید شده:")
print("  1. fig1_utility_convergence.png/pdf")
print("  2. fig2_error_rate.png/pdf")
print("  3. fig3_stability.png/pdf")
print("  4. fig4_pareto_front.png/pdf")
print("  5. fig5_3d_space.png/pdf")
print("  6. fig6_energy_consumption.png/pdf  ⭐ جدید")
print("  7. fig7_delay_performance.png/pdf   ⭐ جدید")
print("  8. fig8_energy_delay_tradeoff.png/pdf  ⭐ جدید")

print("\n📝 جداول LaTeX:")
print("  • table_results.tex (کامل)")
print("  • table_short.tex (مختصر)")

print("\n" + "="*70)
print("🎯 مرحلهٔ بعدی: تولید کدهای TikZ")
print("="*70)
print("\nدستور اجرا:")
print("  python -m analysis.realtime.report_ch5_auto_tikz")
