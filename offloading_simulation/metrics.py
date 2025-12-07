"""
محاسبه معیارهای ارزیابی سیستم Offloading
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class OffloadingResult:
    """نتیجه یک تصمیم Offloading"""
    task_id: int
    layer_name: str
    success: bool
    latency: float  # ms
    energy: float  # Joules
    deadline_met: bool
    processing_time: float
    transmission_time: float


class MetricsCalculator:
    """محاسبه‌گر معیارهای کلیدی"""
    
    @staticmethod
    def calculate_scalability(results: List[OffloadingResult]) -> float:
        """
        مقیاس‌پذیری: درصد Task‌های موفق
        """
        if not results:
            return 0.0
        
        successful = sum(1 for r in results if r.success)
        return (successful / len(results)) * 100
    
    @staticmethod
    def calculate_energy_efficiency(results: List[OffloadingResult]) -> Dict[str, float]:
        """
        کارایی انرژی: میانگین مصرف انرژی
        """
        if not results:
            return {"mean": 0, "std": 0, "total": 0}
        
        energies = [r.energy for r in results if r.success]
        
        if not energies:
            return {"mean": 0, "std": 0, "total": 0}
        
        return {
            "mean": np.mean(energies),
            "std": np.std(energies),
            "total": np.sum(energies),
            "min": np.min(energies),
            "max": np.max(energies)
        }
    
    @staticmethod
    def calculate_latency_reduction(results: List[OffloadingResult],
                                   baseline_layer: str = "ground") -> Dict[str, float]:
        """
        کاهش تاخیر نسبت به لایه مبنا (معمولاً Ground)
        """
        layer_latencies = {}
        
        for result in results:
            if result.success:
                if result.layer_name not in layer_latencies:
                    layer_latencies[result.layer_name] = []
                layer_latencies[result.layer_name].append(result.latency)
        
        # محاسبه میانگین برای هر لایه
        avg_latencies = {
            layer: np.mean(lats) 
            for layer, lats in layer_latencies.items()
        }
        
        if baseline_layer not in avg_latencies:
            return {}
        
        baseline = avg_latencies[baseline_layer]
        
        # محاسبه درصد کاهش
        reductions = {}
        for layer, lat in avg_latencies.items():
            if layer != baseline_layer:
                reduction = ((baseline - lat) / baseline) * 100
                reductions[layer] = reduction
        
        return {
            "baseline": baseline,
            "reductions": reductions,
            "absolute_latencies": avg_latencies
        }
    
    @staticmethod
    def calculate_throughput(results: List[OffloadingResult],
                           time_window: float = 1000.0) -> Dict[str, float]:
        """
        Throughput: تعداد Task در واحد زمان (Task/s)
        
        Args:
            time_window: پنجره زمانی (ms) - پیش‌فرض 1 ثانیه
        """
        successful = [r for r in results if r.success]
        
        if not successful:
            return {"total": 0, "per_layer": {}}
        
        # Throughput کل
        total_throughput = (len(successful) / time_window) * 1000
        
        # Throughput به تفکیک لایه
        layer_throughputs = {}
        layer_counts = {}
        
        for result in successful:
            layer_counts[result.layer_name] = layer_counts.get(result.layer_name, 0) + 1
        
        for layer, count in layer_counts.items():
            layer_throughputs[layer] = (count / time_window) * 1000
        
        return {
            "total": total_throughput,
            "per_layer": layer_throughputs,
            "successful_tasks": len(successful),
            "total_tasks": len(results)
        }
    
    @staticmethod
    def calculate_deadline_compliance(results: List[OffloadingResult]) -> Dict[str, float]:
        """
        میزان رعایت Deadline
        """
        if not results:
            return {"compliance_rate": 0, "violations": 0}
        
        deadline_met = sum(1 for r in results if r.deadline_met and r.success)
        
        return {
            "compliance_rate": (deadline_met / len(results)) * 100,
            "violations": len(results) - deadline_met,
            "total_tasks": len(results)
        }
    
    @staticmethod
    def generate_summary(results: List[OffloadingResult]) -> Dict:
        """تولید خلاصه کامل معیارها"""
        
        calc = MetricsCalculator
        
        return {
            "scalability": calc.calculate_scalability(results),
            "energy_efficiency": calc.calculate_energy_efficiency(results),
            "latency_reduction": calc.calculate_latency_reduction(results),
            "throughput": calc.calculate_throughput(results),
            "deadline_compliance": calc.calculate_deadline_compliance(results),
            "total_results": len(results),
            "successful_results": sum(1 for r in results if r.success)
        }
