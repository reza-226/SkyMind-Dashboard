"""
اجرای آزمایش‌های کامل Offloading
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
from offloading_simulation.layers import LayerFactory
from offloading_simulation.task_generator import TaskGenerator, TaskComplexity
from offloading_simulation.metrics import OffloadingResult, MetricsCalculator
from offloading_simulation.visualizer import OffloadingVisualizer


class OffloadingExperiment:
    """کلاس مدیریت آزمایش"""
    
    def __init__(self, output_dir: str = "results/offloading_results"):
        self.output_dir = Path(output_dir)
        self.layers = LayerFactory.create_all_layers()
        self.task_gen = TaskGenerator()
        self.metrics_calc = MetricsCalculator()
        
    def run_offloading(self, task, layer):
        """شبیه‌سازی یک تصمیم Offloading"""
        
        # محاسبه زمان‌ها
        proc_time = layer.calculate_processing_time(task.computational_load)
        trans_time = layer.calculate_transmission_time(task.data_size)
        total_latency = proc_time + trans_time + layer.base_latency
        
        # محاسبه انرژی
        energy = layer.calculate_energy(task.computational_load)
        
        # بررسی موفقیت
        success = total_latency <= task.deadline
        deadline_met = success
        
        return OffloadingResult(
            task_id=task.task_id,
            layer_name=layer.name.lower(),
            success=success,
            latency=total_latency,
            energy=energy,
            deadline_met=deadline_met,
            processing_time=proc_time,
            transmission_time=trans_time
        )
    
    def run_complexity_experiment(self, complexity: TaskComplexity, num_tasks: int = 100):
        """
        اجرای آزمایش برای یک سطح پیچیدگی
        """
        print(f"\n{'='*60}")
        print(f"🚀 شروع آزمایش: {complexity.value.upper()}")
        print(f"{'='*60}")
        
        # تولید Task‌ها
        tasks = [self.task_gen.generate_task(i, complexity) for i in range(num_tasks)]
        print(f"✅ {num_tasks} Task تولید شد")
        
        # آزمایش روی هر لایه
        layer_results = {}
        
        for layer_name, layer in self.layers.items():
            results = []
            
            for task in tasks:
                result = self.run_offloading(task, layer)
                results.append(result)
            
            # محاسبه معیارها
            scalability = self.metrics_calc.calculate_scalability(results)
            energy_metrics = self.metrics_calc.calculate_energy_efficiency(results)
            throughput = self.metrics_calc.calculate_throughput(results)
            
            avg_latency = np.mean([r.latency for r in results if r.success]) if results else 0
            
            layer_results[layer_name] = {
                "scalability": scalability,
                "energy_mean": energy_metrics["mean"],
                "energy_std": energy_metrics["std"],
                "throughput": throughput["total"],
                "avg_latency": avg_latency,
                "raw_results": [
                    {
                        "task_id": r.task_id,
                        "success": r.success,
                        "latency": r.latency,
                        "energy": r.energy
                    }
                    for r in results
                ]
            }
            
            print(f"  {layer.name:8s} | Success: {scalability:5.1f}% | "
                  f"Energy: {energy_metrics['mean']:6.2f}J | "
                  f"Latency: {avg_latency:7.2f}ms")
        
        return layer_results
    
    def save_results(self, results: dict, complexity: str):
        """ذخیره نتایج"""
        output_path = self.output_dir / complexity
        output_path.mkdir(parents=True, exist_ok=True)
        
        # ذخیره JSON
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 نتایج ذخیره شد: {metrics_file}")
    
    def run_all_experiments(self):
        """اجرای همه آزمایش‌ها"""
        print("\n" + "="*60)
        print("🎯 شروع آزمایش‌های کامل Offloading")
        print("="*60)
        
        complexities = [
            (TaskComplexity.SIMPLE, "simple"),
            (TaskComplexity.MEDIUM, "medium"),
            (TaskComplexity.COMPLEX, "complex")
        ]
        
        for complexity_enum, complexity_name in complexities:
            results = self.run_complexity_experiment(complexity_enum, num_tasks=100)
            self.save_results(results, complexity_name)
        
        print("\n" + "="*60)
        print("✅ همه آزمایش‌ها با موفقیت اجرا شدند!")
        print("="*60)
        
        # تولید نمودارها
        print("\n🎨 تولید نمودارها...")
        visualizer = OffloadingVisualizer()
        visualizer.generate_all_visualizations(str(self.output_dir))


if __name__ == "__main__":
    experiment = OffloadingExperiment()
    experiment.run_all_experiments()
