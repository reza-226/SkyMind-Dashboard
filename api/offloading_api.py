"""
API برای فراخوانی نتایج Offloading در Dashboard
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class OffloadingAPI:
    """API دسترسی به نتایج"""
    
    def __init__(self, results_dir: str = "results/offloading_results"):
        self.results_dir = Path(results_dir)
    
    def get_complexity_results(self, complexity: str) -> Optional[Dict]:
        """
        دریافت نتایج یک سطح پیچیدگی
        
        Args:
            complexity: 'simple', 'medium', 'complex'
        """
        metrics_file = self.results_dir / complexity / "metrics.json"
        
        if not metrics_file.exists():
            return None
        
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def get_all_results(self) -> Dict:
        """دریافت همه نتایج"""
        results = {}
        
        for complexity in ['simple', 'medium', 'complex']:
            data = self.get_complexity_results(complexity)
            if data:
                results[complexity] = data
        
        return results
    
    def get_best_layer_for_metric(self, metric: str, complexity: str = None) -> Dict:
        """
        یافتن بهترین لایه برای یک معیار خاص
        
        Args:
            metric: 'scalability', 'energy_mean', 'throughput', 'avg_latency'
            complexity: اختیاری - اگر None، همه سطوح را بررسی می‌کند
        """
        if complexity:
            data = self.get_complexity_results(complexity)
            if not data:
                return {}
            
            results = {complexity: self._find_best_layer(data, metric)}
        else:
            all_data = self.get_all_results()
            results = {}
            
            for comp, data in all_data.items():
                results[comp] = self._find_best_layer(data, metric)
        
        return results
    
    def _find_best_layer(self, data: Dict, metric: str) -> Dict:
        """یافتن بهترین لایه برای یک معیار"""
        
        # برای Energy و Latency، کمترین بهتر است
        minimize_metrics = ['energy_mean', 'avg_latency']
        
        best_layer = None
        best_value = float('inf') if metric in minimize_metrics else float('-inf')
        
        for layer, metrics in data.items():
            if metric in metrics:
                value = metrics[metric]
                
                if metric in minimize_metrics:
                    if value < best_value:
                        best_value = value
                        best_layer = layer
                else:
                    if value > best_value:
                        best_value = value
                        best_layer = layer
        
        return {
            "best_layer": best_layer,
            "value": best_value,
            "metric": metric
        }
    
    def get_summary_statistics(self) -> Dict:
        """دریافت آمار خلاصه"""
        all_results = self.get_all_results()
        
        summary = {
            "total_complexities": len(all_results),
            "layers_tested": list(next(iter(all_results.values())).keys()) if all_results else [],
            "best_performers": {}
        }
        
        # بهترین لایه برای هر معیار
        metrics = ['scalability', 'energy_mean', 'throughput', 'avg_latency']
        
        for metric in metrics:
            summary["best_performers"][metric] = self.get_best_layer_for_metric(metric)
        
        return summary
    
    def export_for_dashboard(self, output_file: str = "dashboard_data.json"):
        """صادرات داده برای Dashboard"""
        
        dashboard_data = {
            "all_results": self.get_all_results(),
            "summary": self.get_summary_statistics(),
            "charts_paths": {
                "simple": {
                    "radar": "results/offloading_results/simple/charts/radar_chart.png",
                    "bars": "results/offloading_results/simple/charts/comparison_bars.png"
                },
                "medium": {
                    "radar": "results/offloading_results/medium/charts/radar_chart.png",
                    "bars": "results/offloading_results/medium/charts/comparison_bars.png"
                },
                "complex": {
                    "radar": "results/offloading_results/complex/charts/radar_chart.png",
                    "bars": "results/offloading_results/complex/charts/comparison_bars.png"
                },
                "heatmap": "results/offloading_results/visualizations/scalability_heatmap.png"
            }
        }
        
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        print(f"✅ Dashboard data exported: {output_path}")
        return dashboard_data


# تست API
if __name__ == "__main__":
    api = OffloadingAPI()
    
    print("🔍 دریافت نتایج...\n")
    
    # نمایش خلاصه
    summary = api.get_summary_statistics()
    print(json.dumps(summary, indent=2))
    
    # صادرات برای Dashboard
    api.export_for_dashboard()
