"""
generate_pareto_front.py
تحلیل Trade-off بین Delay و Energy + محاسبه Pareto Front
بارگذاری خودکار از آخرین فایل JSON مقایسه + ذخیره‌سازی
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

def load_latest_comparison_results():
    """بارگذاری آخرین فایل JSON از مقایسه"""
    comparison_dir = Path("results/comparison")
    
    if not comparison_dir.exists():
        raise FileNotFoundError(f"❌ Directory not found: {comparison_dir}")
    
    json_files = list(comparison_dir.glob("comparison_*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"❌ No comparison JSON files found in {comparison_dir}")
    
    # انتخاب آخرین فایل بر اساس timestamp
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    
    print(f"📂 Loading results from: {latest_file.name}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def extract_pareto_data(data):
    """استخراج داده‌های Delay و Energy از نتایج"""
    results = data['results']
    policies = data['experiment_info']['policies']
    
    pareto_data = {}
    
    for policy in policies:
        delay_mean = results[policy]['delays']['mean']
        energy_mean = results[policy]['energies']['mean']
        pareto_data[policy] = {
            'delay': delay_mean,
            'energy': energy_mean
        }
    
    return pareto_data

def is_pareto_efficient(costs):
    """
    محاسبه نقاط Pareto-optimal
    costs: آرایه Nx2 (هر سطر: [delay, energy])
    برمی‌گرداند: آرایه boolean (True = Pareto-optimal)
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # یک نقطه Pareto است اگر هیچ نقطه‌ای در هر دو بعد بهتر نباشد
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    
    return is_efficient

def compute_pareto_front(pareto_data):
    """محاسبه Pareto Front از داده‌ها"""
    policies = list(pareto_data.keys())
    costs = np.array([[pareto_data[p]['delay'], pareto_data[p]['energy']] 
                      for p in policies])
    
    pareto_mask = is_pareto_efficient(costs)
    
    pareto_policies = [policies[i] for i in range(len(policies)) if pareto_mask[i]]
    dominated_policies = [policies[i] for i in range(len(policies)) if not pareto_mask[i]]
    
    return pareto_policies, dominated_policies, pareto_mask

def plot_pareto_front(pareto_data, pareto_policies, dominated_policies, save_dir):
    """رسم نمودار Pareto Front"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # رنگ‌بندی
    color_pareto = '#E74C3C'      # قرمز برای Pareto
    color_dominated = '#3498DB'   # آبی برای Dominated
    
    # رسم نقاط Dominated
    for policy in dominated_policies:
        delay = pareto_data[policy]['delay']
        energy = pareto_data[policy]['energy'] / 1e5  # تبدیل به ×10⁵
        
        ax.scatter(delay, energy, s=150, marker='o', 
                   color=color_dominated, alpha=0.7, 
                   edgecolors='black', linewidths=2,
                   label='Dominated' if policy == dominated_policies[0] else '')
        
        ax.annotate(policy.replace('_', ' ').title(), 
                    (delay, energy),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor=color_dominated, alpha=0.3))
    
    # رسم نقاط Pareto
    pareto_delays = []
    pareto_energies = []
    
    for policy in pareto_policies:
        delay = pareto_data[policy]['delay']
        energy = pareto_data[policy]['energy'] / 1e5
        
        pareto_delays.append(delay)
        pareto_energies.append(energy)
        
        ax.scatter(delay, energy, s=300, marker='*', 
                   color=color_pareto, alpha=0.9,
                   edgecolors='black', linewidths=2,
                   label='Pareto Optimal' if policy == pareto_policies[0] else '',
                   zorder=5)
        
        ax.annotate(policy.replace('_', ' ').title(), 
                    (delay, energy),
                    xytext=(10, -15), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor=color_pareto, alpha=0.3))
    
    # رسم خط Pareto Front (اگر بیش از یک نقطه داشته باشیم)
    if len(pareto_delays) > 1:
        # مرتب‌سازی بر اساس delay
        sorted_indices = np.argsort(pareto_delays)
        sorted_delays = np.array(pareto_delays)[sorted_indices]
        sorted_energies = np.array(pareto_energies)[sorted_indices]
        
        ax.plot(sorted_delays, sorted_energies, 
                color=color_pareto, linestyle='--', 
                linewidth=2, alpha=0.6, zorder=3,
                label='Pareto Front')
    
    # تنظیمات محورها
    ax.set_xlabel('Average Delay (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Energy (×10⁵ J)', fontsize=14, fontweight='bold')
    ax.set_title('Delay-Energy Pareto Front Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # اضافه کردن توضیحات
    textstr = '\n'.join([
        'Pareto-Optimal Points:',
        '• No other policy dominates them',
        '• in both Delay AND Energy',
        '',
        'Dominated Points:',
        '• At least one Pareto point',
        '• is better in both metrics'
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    return fig

# ============================================================================
# اجرای اصلی
# ============================================================================
print("=" * 70)
print("📈 Pareto Front Analysis")
print("=" * 70)

try:
    # بارگذاری نتایج
    data = load_latest_comparison_results()
    
    # استخراج داده‌ها
    pareto_data = extract_pareto_data(data)
    
    print("\n📊 Extracted Data:")
    print("-" * 70)
    for policy, values in pareto_data.items():
        print(f"   {policy:20s}: Delay = {values['delay']:8.4f}s, "
              f"Energy = {values['energy']:12.4e}J")
    
    # محاسبه Pareto Front
    pareto_policies, dominated_policies, pareto_mask = compute_pareto_front(pareto_data)
    
    print("\n" + "=" * 70)
    print("🎯 Pareto Analysis Results")
    print("=" * 70)
    print(f"\n✅ Pareto-Optimal Policies ({len(pareto_policies)}):")
    for p in pareto_policies:
        print(f"   • {p}")
    
    print(f"\n❌ Dominated Policies ({len(dominated_policies)}):")
    for p in dominated_policies:
        print(f"   • {p}")
    
    # ذخیره‌سازی
    results_dir = Path("results/pareto")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. ذخیره JSON
    json_path = results_dir / f"pareto_analysis_{timestamp}.json"
    output_data = {
        'timestamp': timestamp,
        'source_file': data['experiment_info']['timestamp'],
        'pareto_data': pareto_data,
        'pareto_optimal_policies': pareto_policies,
        'dominated_policies': dominated_policies,
        'analysis': {
            'n_total_policies': len(pareto_data),
            'n_pareto_optimal': len(pareto_policies),
            'n_dominated': len(dominated_policies)
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ JSON saved: {json_path}")
    
    # 2. رسم و ذخیره نمودار
    fig = plot_pareto_front(pareto_data, pareto_policies, dominated_policies, results_dir)
    
    png_path = results_dir / f"pareto_front_{timestamp}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✅ PNG saved: {png_path}")
    
    pdf_path = results_dir / f"pareto_front_{timestamp}.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"✅ PDF saved: {pdf_path}")
    
    plt.show()
    
    # 3. ذخیره گزارش متنی
    txt_path = results_dir / f"pareto_report_{timestamp}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Pareto Front Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source Data: comparison_{data['experiment_info']['timestamp']}.json\n")
        f.write(f"Total Policies Analyzed: {len(pareto_data)}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Raw Data\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Policy':<20} {'Delay (s)':<15} {'Energy (J)':<15}\n")
        f.write("-" * 50 + "\n")
        for policy, values in pareto_data.items():
            f.write(f"{policy:<20} {values['delay']:<15.4f} {values['energy']:<15.4e}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Pareto Analysis\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Pareto-Optimal Policies ({len(pareto_policies)}):\n")
        f.write("-" * 40 + "\n")
        for p in pareto_policies:
            f.write(f"  • {p}\n")
            f.write(f"    Delay:  {pareto_data[p]['delay']:.4f} s\n")
            f.write(f"    Energy: {pareto_data[p]['energy']:.4e} J\n\n")
        
        f.write(f"Dominated Policies ({len(dominated_policies)}):\n")
        f.write("-" * 40 + "\n")
        for p in dominated_policies:
            f.write(f"  • {p}\n")
            f.write(f"    Delay:  {pareto_data[p]['delay']:.4f} s\n")
            f.write(f"    Energy: {pareto_data[p]['energy']:.4e} J\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("Interpretation\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Pareto-Optimal points represent policies where:\n")
        f.write("  • No other policy is better in BOTH delay AND energy\n")
        f.write("  • Any improvement in one metric causes degradation in the other\n\n")
        
        f.write("Dominated points represent policies where:\n")
        f.write("  • At least one Pareto-optimal policy outperforms them\n")
        f.write("  • in both delay and energy simultaneously\n")
    
    print(f"✅ Text report saved: {txt_path}")
    
    print("\n" + "=" * 70)
    print("✅ Pareto analysis completed successfully!")
    print(f"📁 All results saved in: {results_dir}")
    print("=" * 70)

except FileNotFoundError as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 Please run 'compare_all_policies.py' first to generate comparison data.")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
