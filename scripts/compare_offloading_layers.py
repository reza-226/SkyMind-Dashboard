#!/usr/bin/env python3
"""
مقایسه تخلیه محاسباتی در چهار سطح: Ground, Edge, Fog, Cloud
نسخه اصلاح‌شده (رفع خطای JSON)
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
import json
import os

@dataclass
class OffloadingLayer:
    """تعریف یک لایه offloading"""
    name: str
    latency_ms: float  # میلی‌ثانیه
    compute_capacity: float  # GHz
    energy_cost: float  # Joule
    bandwidth_mbps: float  # Mbps
    distance_km: float  # کیلومتر
    reliability: float  # 0-1
    cost_per_task: float  # دلار

class FourLayerOffloadingComparison:
    """مقایسه offloading در چهار لایه"""
    
    def __init__(self):
        self.layers = {
            'Ground': OffloadingLayer(
                name='Ground (محلی)',
                latency_ms=0.5,
                compute_capacity=1.0,  # GHz
                energy_cost=10.0,  # J
                bandwidth_mbps=float('inf'),
                distance_km=0.0,
                reliability=0.70,
                cost_per_task=0.0
            ),
            'Edge': OffloadingLayer(
                name='Edge (لبه)',
                latency_ms=5.0,
                compute_capacity=5.0,
                energy_cost=3.0,
                bandwidth_mbps=100.0,
                distance_km=0.5,
                reliability=0.85,
                cost_per_task=0.01
            ),
            'Fog': OffloadingLayer(
                name='Fog (مه)',
                latency_ms=25.0,
                compute_capacity=20.0,
                energy_cost=1.0,
                bandwidth_mbps=500.0,
                distance_km=5.0,
                reliability=0.95,
                cost_per_task=0.05
            ),
            'Cloud': OffloadingLayer(
                name='Cloud (ابر)',
                latency_ms=80.0,
                compute_capacity=100.0,
                energy_cost=0.2,
                bandwidth_mbps=1000.0,
                distance_km=100.0,
                reliability=0.99,
                cost_per_task=0.10
            )
        }
    
    def calculate_total_delay(self, layer: OffloadingLayer, 
                             task_size_mb: float) -> float:
        """محاسبه تأخیر کل"""
        if layer.bandwidth_mbps == float('inf'):
            transmission_delay = 0
        else:
            transmission_delay = (task_size_mb / layer.bandwidth_mbps) * 1000
        
        computation_delay = 100 / layer.compute_capacity
        total_delay = layer.latency_ms + transmission_delay + computation_delay
        
        return total_delay
    
    def calculate_energy(self, layer: OffloadingLayer,
                        task_size_mb: float) -> float:
        """محاسبه مصرف انرژی"""
        transmission_energy = task_size_mb * 0.1  # J/MB
        total_energy = layer.energy_cost + transmission_energy
        return total_energy
    
    def evaluate_layers(self, task_sizes: List[float]) -> pd.DataFrame:
        """ارزیابی همه لایه‌ها"""
        results = []
        
        for task_size in task_sizes:
            for name, layer in self.layers.items():
                delay = self.calculate_total_delay(layer, task_size)
                energy = self.calculate_energy(layer, task_size)
                cost = layer.cost_per_task * task_size
                
                results.append({
                    'Layer': name,
                    'Task_Size_MB': task_size,
                    'Total_Delay_ms': delay,
                    'Energy_J': energy,
                    'Cost_USD': cost,
                    'Reliability': layer.reliability,
                    'Distance_km': layer.distance_km
                })
        
        return pd.DataFrame(results)
    
    def plot_comparison(self, df: pd.DataFrame, output_dir: str = 'results/offloading_comparison'):
        """رسم نمودارهای مقایسه"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('مقایسه چهار سطح Offloading: Ground, Edge, Fog, Cloud', 
                     fontsize=16, fontweight='bold')
        
        # 1. تأخیر vs اندازه Task
        for layer in df['Layer'].unique():
            data = df[df['Layer'] == layer]
            axes[0, 0].plot(data['Task_Size_MB'], data['Total_Delay_ms'], 
                          marker='o', label=layer, linewidth=2)
        axes[0, 0].set_xlabel('Task Size (MB)')
        axes[0, 0].set_ylabel('Total Delay (ms)')
        axes[0, 0].set_title('Latency Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. مصرف انرژی vs اندازه Task
        for layer in df['Layer'].unique():
            data = df[df['Layer'] == layer]
            axes[0, 1].plot(data['Task_Size_MB'], data['Energy_J'], 
                          marker='s', label=layer, linewidth=2)
        axes[0, 1].set_xlabel('Task Size (MB)')
        axes[0, 1].set_ylabel('Energy Consumption (J)')
        axes[0, 1].set_title('Energy Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. هزینه vs اندازه Task
        for layer in df['Layer'].unique():
            data = df[df['Layer'] == layer]
            axes[1, 0].plot(data['Task_Size_MB'], data['Cost_USD'], 
                          marker='^', label=layer, linewidth=2)
        axes[1, 0].set_xlabel('Task Size (MB)')
        axes[1, 0].set_ylabel('Cost (USD)')
        axes[1, 0].set_title('Cost Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Radar Chart
        categories = ['Latency', 'Energy', 'Cost', 'Reliability']
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax_radar = plt.subplot(2, 2, 4, projection='polar')
        
        df_10mb = df[df['Task_Size_MB'] == 10.0]
        
        for layer in df_10mb['Layer'].unique():
            data = df_10mb[df_10mb['Layer'] == layer].iloc[0]
            values = [
                1 - (data['Total_Delay_ms'] / 200),
                1 - (data['Energy_J'] / 20),
                1 - (data['Cost_USD'] / 2),
                data['Reliability']
            ]
            values += values[:1]
            ax_radar.plot(angles, values, marker='o', label=layer, linewidth=2)
            ax_radar.fill(angles, values, alpha=0.15)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(categories)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Multi-Dimensional Comparison (10MB Task)')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax_radar.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/four_layer_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ نمودار ذخیره شد: {output_dir}/four_layer_comparison.png")
        plt.show()
    
    def generate_report(self, df: pd.DataFrame, output_dir: str = 'results/offloading_comparison'):
        """تولید گزارش کامل - نسخه اصلاح‌شده"""
        os.makedirs(output_dir, exist_ok=True)
        
        # خلاصه آماری
        summary = df.groupby('Layer').agg({
            'Total_Delay_ms': ['mean', 'std'],
            'Energy_J': ['mean', 'std'],
            'Cost_USD': ['mean', 'std'],
            'Reliability': 'first'
        }).round(3)
        
        print("\n" + "="*70)
        print("📊 خلاصه آماری مقایسه Offloading")
        print("="*70)
        print(summary)
        
        # ذخیره به CSV
        df.to_csv(f'{output_dir}/offloading_results.csv', index=False)
        summary.to_csv(f'{output_dir}/offloading_summary.csv')
        
        # ✅ رفع خطای JSON - تبدیل صحیح summary
        results_dict = {
            'layers': {},
            'summary': {}
        }
        
        # تبدیل summary به فرمت قابل JSON
        for layer in summary.index:
            results_dict['summary'][layer] = {
                'total_delay': {
                    'mean': float(summary.loc[layer, ('Total_Delay_ms', 'mean')]),
                    'std': float(summary.loc[layer, ('Total_Delay_ms', 'std')])
                },
                'energy': {
                    'mean': float(summary.loc[layer, ('Energy_J', 'mean')]),
                    'std': float(summary.loc[layer, ('Energy_J', 'std')])
                },
                'cost': {
                    'mean': float(summary.loc[layer, ('Cost_USD', 'mean')]),
                    'std': float(summary.loc[layer, ('Cost_USD', 'std')])
                },
                'reliability': float(summary.loc[layer, ('Reliability', 'first')])
            }
        
        # اطلاعات لایه‌ها
        for layer_name, layer in self.layers.items():
            results_dict['layers'][layer_name] = {
                'latency_ms': layer.latency_ms,
                'compute_capacity_GHz': layer.compute_capacity,
                'energy_cost_J': layer.energy_cost,
                'bandwidth_Mbps': 'unlimited' if layer.bandwidth_mbps == float('inf') else layer.bandwidth_mbps,
                'distance_km': layer.distance_km,
                'reliability': layer.reliability,
                'cost_per_task_USD': layer.cost_per_task
            }
        
        with open(f'{output_dir}/offloading_config.json', 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ نتایج ذخیره شد در: {output_dir}/")
        print(f"   - offloading_results.csv")
        print(f"   - offloading_summary.csv")
        print(f"   - offloading_config.json")
        print(f"   - four_layer_comparison.png")

def main():
    """اجرای مقایسه"""
    print("="*70)
    print("🚀 شروع مقایسه Offloading در چهار سطح")
    print("="*70)
    
    comparison = FourLayerOffloadingComparison()
    task_sizes = [1, 5, 10, 20, 50, 100]
    
    df = comparison.evaluate_layers(task_sizes)
    comparison.plot_comparison(df)
    comparison.generate_report(df)
    
    print("\n" + "="*70)
    print("✅ مقایسه با موفقیت انجام شد!")
    print("="*70)

if __name__ == "__main__":
    main()
