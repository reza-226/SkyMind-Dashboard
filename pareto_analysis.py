"""
pareto_analysis.py
==================
تحلیل کامل Pareto Front برای نتایج آزمایش‌ها
- Pareto Dominance Analysis
- Non-dominated Solutions
- Hypervolume Indicator
- Spread Metric
- Spacing Metric
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict

# تنظیمات رسم
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

class ParetoAnalyzer:
    """تحلیلگر Pareto Front"""
    
    def __init__(self, results_path='results/obstacle_experiments_fixed.json'):
        """بارگذاری نتایج"""
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # استخراج نقاط (Energy, Delay)
        self.points = {}
        self.policies = list(self.results.keys())
        
        for policy in self.policies:
            energy = self.results[policy]['mean_energy']
            delay = self.results[policy]['mean_delay']
            self.points[policy] = (energy, delay)
        
        print("✅ نتایج بارگذاری شد:")
        for policy, (e, d) in self.points.items():
            print(f"  - {policy}: Energy={e:.2e}, Delay={d:.2f}s")
    
    def is_dominated(self, point1: Tuple[float, float], 
                     point2: Tuple[float, float]) -> bool:
        """
        بررسی اینکه آیا point1 توسط point2 dominated می‌شود
        (هر دو هدف کمینه‌سازی هستند)
        
        point1 dominated است اگر:
        - Energy(point2) <= Energy(point1) AND Delay(point2) <= Delay(point1)
        - حداقل یکی از نامساوی‌ها strict باشد
        """
        e1, d1 = point1
        e2, d2 = point2
        
        # point2 باید در هر دو هدف بهتر یا مساوی باشد
        better_or_equal = (e2 <= e1) and (d2 <= d1)
        # حداقل در یکی بهتر باشد (نه فقط مساوی)
        strictly_better = (e2 < e1) or (d2 < d1)
        
        return better_or_equal and strictly_better
    
    def find_pareto_optimal(self) -> Dict[str, bool]:
        """
        پیدا کردن سیاست‌های Pareto-optimal
        (سیاست‌هایی که توسط هیچ سیاست دیگری dominated نمی‌شوند)
        """
        pareto_optimal = {}
        
        for policy1 in self.policies:
            is_dominated_flag = False
            for policy2 in self.policies:
                if policy1 != policy2:
                    if self.is_dominated(self.points[policy1], 
                                        self.points[policy2]):
                        is_dominated_flag = True
                        break
            pareto_optimal[policy1] = not is_dominated_flag
        
        return pareto_optimal
    
    def dominance_matrix(self) -> np.ndarray:
        """
        ماتریس Dominance
        matrix[i][j] = 1 اگر سیاست i توسط سیاست j dominated شود
        """
        n = len(self.policies)
        matrix = np.zeros((n, n), dtype=int)
        
        for i, policy1 in enumerate(self.policies):
            for j, policy2 in enumerate(self.policies):
                if i != j:
                    if self.is_dominated(self.points[policy1], 
                                        self.points[policy2]):
                        matrix[i][j] = 1
        
        return matrix
    
    def hypervolume(self, reference_point: Tuple[float, float] = None) -> float:
        """
        محاسبه Hypervolume Indicator
        
        Hypervolume = حجم فضای هدف که توسط راه‌حل‌ها dominated می‌شود
        (نسبت به یک نقطه مرجع)
        
        برای 2D: مساحت زیر Pareto Front
        """
        if reference_point is None:
            # نقطه مرجع: بدترین مقادیر + margin
            max_energy = max(p[0] for p in self.points.values())
            max_delay = max(p[1] for p in self.points.values())
            reference_point = (max_energy * 1.1, max_delay * 1.1)
        
        ref_e, ref_d = reference_point
        
        # مرتب‌سازی نقاط بر اساس Energy
        sorted_points = sorted(self.points.values(), key=lambda x: x[0])
        
        # محاسبه مساحت با روش ترپزوئیدی
        hv = 0.0
        prev_e = 0.0
        
        for e, d in sorted_points:
            # مساحت مستطیل
            width = e - prev_e
            height = ref_d - d
            hv += width * height
            prev_e = e
        
        # آخرین مستطیل تا نقطه مرجع
        last_e, last_d = sorted_points[-1]
        hv += (ref_e - last_e) * (ref_d - last_d)
        
        return hv
    
    def spread_metric(self) -> float:
        """
        محاسبه Spread Metric (متریک پراکندگی)
        
        اندازه‌گیری توزیع یکنواخت راه‌حل‌ها روی Pareto Front
        مقدار کمتر = توزیع بهتر
        
        Δ = (d_f + d_l + Σ|d_i - d̄|) / (d_f + d_l + (N-1)d̄)
        """
        # مرتب‌سازی بر اساس Energy
        sorted_points = sorted(self.points.values(), key=lambda x: x[0])
        
        if len(sorted_points) < 2:
            return 0.0
        
        # محاسبه فاصله اقلیدسی بین نقاط متوالی
        distances = []
        for i in range(len(sorted_points) - 1):
            e1, d1 = sorted_points[i]
            e2, d2 = sorted_points[i + 1]
            dist = np.sqrt((e2 - e1)**2 + (d2 - d1)**2)
            distances.append(dist)
        
        # d_f: فاصله از نقطه اول تا Ideal Point (0, 0)
        e_first, d_first = sorted_points[0]
        d_f = np.sqrt(e_first**2 + d_first**2)
        
        # d_l: فاصله از نقطه آخر تا Nadir Point (max_e, max_d)
        e_last, d_last = sorted_points[-1]
        max_e = max(p[0] for p in self.points.values())
        max_d = max(p[1] for p in self.points.values())
        d_l = np.sqrt((max_e - e_last)**2 + (max_d - d_last)**2)
        
        # میانگین فاصله‌ها
        d_mean = np.mean(distances)
        
        # Spread Metric
        numerator = d_f + d_l + sum(abs(d - d_mean) for d in distances)
        denominator = d_f + d_l + (len(sorted_points) - 1) * d_mean
        
        spread = numerator / denominator if denominator > 0 else 0.0
        
        return spread
    
    def spacing_metric(self) -> float:
        """
        محاسبه Spacing Metric
        
        اندازه‌گیری یکنواختی فاصله بین راه‌حل‌های متوالی
        مقدار کمتر = فاصله‌گذاری یکنواخت‌تر
        
        S = √(1/(N-1) Σ(d_i - d̄)²)
        """
        sorted_points = sorted(self.points.values(), key=lambda x: x[0])
        
        if len(sorted_points) < 2:
            return 0.0
        
        # فاصله‌های متوالی
        distances = []
        for i in range(len(sorted_points) - 1):
            e1, d1 = sorted_points[i]
            e2, d2 = sorted_points[i + 1]
            dist = np.sqrt((e2 - e1)**2 + (d2 - d1)**2)
            distances.append(dist)
        
        # میانگین و انحراف معیار
        d_mean = np.mean(distances)
        spacing = np.sqrt(np.mean([(d - d_mean)**2 for d in distances]))
        
        return spacing
    
    def plot_pareto_analysis(self, output_dir='results/plots'):
        """رسم تحلیل Pareto با جزئیات کامل"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # پیدا کردن سیاست‌های Pareto-optimal
        pareto_optimal = self.find_pareto_optimal()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = {'Random': '#FF6B6B', 'Greedy': '#4ECDC4', 
                  'Obstacle-Aware': '#45B7D1', 'Hybrid': '#FFA07A'}
        markers = {'Random': 'o', 'Greedy': 's', 
                   'Obstacle-Aware': '^', 'Hybrid': 'D'}
        
        # رسم نقاط
        for policy in self.policies:
            e, d = self.points[policy]
            is_pareto = pareto_optimal[policy]
            
            # اندازه و ضخامت بیشتر برای Pareto-optimal
            size = 300 if is_pareto else 200
            edge_width = 3 if is_pareto else 2
            
            ax.scatter(e, d, s=size, color=colors[policy], 
                      marker=markers[policy], edgecolor='black',
                      linewidth=edge_width, alpha=0.8,
                      label=f"{policy} {'⭐' if is_pareto else ''}")
            
            # برچسب
            offset_x = 5e3 if policy != 'Random' else -5e3
            offset_y = 1 if policy in ['Greedy', 'Random'] else -1
            ax.annotate(policy, (e, d),
                       textcoords="offset points",
                       xytext=(offset_x, offset_y),
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5',
                                facecolor='yellow' if is_pareto else 'white',
                                alpha=0.7))
        
        # رسم Pareto Front (خط اتصال سیاست‌های Pareto-optimal)
        pareto_points = [self.points[p] for p in self.policies 
                        if pareto_optimal[p]]
        if len(pareto_points) >= 2:
            pareto_sorted = sorted(pareto_points, key=lambda x: x[0])
            energies = [p[0] for p in pareto_sorted]
            delays = [p[1] for p in pareto_sorted]
            ax.plot(energies, delays, 'k--', linewidth=2, 
                   alpha=0.5, label='Pareto Front')
        
        ax.set_xlabel('Energy Consumption (Joules)', 
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Delay (seconds)', 
                     fontsize=13, fontweight='bold')
        ax.set_title('Pareto Front Analysis\n(⭐ = Pareto-Optimal Solutions)',
                    fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        ax.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pareto_analysis_detailed.png', dpi=300)
        plt.close()
        
        print(f"✅ رسم شد: {output_dir}/pareto_analysis_detailed.png")
    
    def generate_report(self, output_file='results/pareto_report.txt'):
        """تولید گزارش متنی کامل"""
        pareto_optimal = self.find_pareto_optimal()
        dom_matrix = self.dominance_matrix()
        hv = self.hypervolume()
        spread = self.spread_metric()
        spacing = self.spacing_metric()
        
        report = []
        report.append("="*70)
        report.append(" 📊 PARETO FRONT ANALYSIS REPORT")
        report.append("="*70)
        report.append("")
        
        # 1. Pareto-Optimal Solutions
        report.append("🌟 1. PARETO-OPTIMAL SOLUTIONS:")
        report.append("-" * 70)
        for policy in self.policies:
            status = "✅ Pareto-Optimal" if pareto_optimal[policy] else "❌ Dominated"
            e, d = self.points[policy]
            report.append(f"  {policy:20s} | {status:20s} | "
                         f"E={e:.2e}, D={d:.2f}s")
        report.append("")
        
        # 2. Dominance Matrix
        report.append("📋 2. DOMINANCE MATRIX:")
        report.append("-" * 70)
        report.append("  (Row i is dominated by Column j if matrix[i][j] = 1)")
        report.append("")
        header = "     " + " ".join(f"{p[:4]:>5s}" for p in self.policies)
        report.append(header)
        for i, policy in enumerate(self.policies):
            row = f"{policy[:4]:>5s}" + " ".join(f"{dom_matrix[i][j]:>5d}" 
                                                  for j in range(len(self.policies)))
            report.append(row)
        report.append("")
        
        # 3. Dominance Relations
        report.append("🔗 3. DOMINANCE RELATIONS:")
        report.append("-" * 70)
        for i, policy1 in enumerate(self.policies):
            dominates = [self.policies[j] for j in range(len(self.policies))
                        if dom_matrix[j][i] == 1]
            dominated_by = [self.policies[j] for j in range(len(self.policies))
                           if dom_matrix[i][j] == 1]
            
            report.append(f"  {policy1}:")
            if dominates:
                report.append(f"    ✓ Dominates: {', '.join(dominates)}")
            if dominated_by:
                report.append(f"    ✗ Dominated by: {', '.join(dominated_by)}")
            if not dominates and not dominated_by:
                report.append(f"    ⚖️  No dominance relations")
        report.append("")
        
        # 4. Quality Metrics
        report.append("📈 4. QUALITY METRICS:")
        report.append("-" * 70)
        report.append(f"  Hypervolume (HV):        {hv:.4e}")
        report.append(f"    → Higher is better (larger dominated space)")
        report.append(f"  Spread Metric (Δ):       {spread:.6f}")
        report.append(f"    → Lower is better (uniform distribution)")
        report.append(f"  Spacing Metric (S):      {spacing:.6f}")
        report.append(f"    → Lower is better (uniform spacing)")
        report.append("")
        
        # 5. Recommendations
        report.append("💡 5. RECOMMENDATIONS:")
        report.append("-" * 70)
        
        # بهترین برای Delay
        best_delay = min(self.policies, key=lambda p: self.points[p][1])
        e_bd, d_bd = self.points[best_delay]
        report.append(f"  🚀 Best for Delay:       {best_delay} "
                     f"(D={d_bd:.2f}s, E={e_bd:.2e}J)")
        
        # بهترین برای Energy
        best_energy = min(self.policies, key=lambda p: self.points[p][0])
        e_be, d_be = self.points[best_energy]
        report.append(f"  ⚡ Best for Energy:      {best_energy} "
                     f"(E={e_be:.2e}J, D={d_be:.2f}s)")
        
        # متعادل‌ترین Pareto-optimal
        pareto_policies = [p for p in self.policies if pareto_optimal[p]]
        if len(pareto_policies) > 2:
            # نرمال‌سازی و پیدا کردن نزدیک‌ترین به (0.5, 0.5)
            max_e = max(self.points[p][0] for p in pareto_policies)
            max_d = max(self.points[p][1] for p in pareto_policies)
            
            balanced = min(pareto_policies, 
                          key=lambda p: (self.points[p][0]/max_e - 0.5)**2 + 
                                       (self.points[p][1]/max_d - 0.5)**2)
            e_bal, d_bal = self.points[balanced]
            report.append(f"  ⚖️  Most Balanced:        {balanced} "
                         f"(E={e_bal:.2e}J, D={d_bal:.2f}s)")
        
        report.append("")
        report.append("="*70)
        
        # ذخیره و چاپ
        report_text = "\n".join(report)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✅ گزارش ذخیره شد: {output_file}")


def main():
    """اجرای تحلیل کامل"""
    print("🔄 شروع تحلیل Pareto Front...\n")
    
    analyzer = ParetoAnalyzer()
    
    print("\n📊 در حال رسم نمودار تحلیل...")
    analyzer.plot_pareto_analysis()
    
    print("\n📝 در حال تولید گزارش...")
    analyzer.generate_report()
    
    print("\n✅ تحلیل Pareto Front کامل شد!")


if __name__ == "__main__":
    main()
