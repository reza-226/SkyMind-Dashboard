"""
اجرای آزمایش‌های چندسطحی و ذخیره نتایج
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from .scenario_loader import ScenarioLoader, Scenario


class ExperimentRunner:
    """مدیریت و اجرای آزمایش‌های Multi-Tier"""
    
    def __init__(self, config_path: str = "experiments/scenarios_config.yaml"):
        self.loader = ScenarioLoader(config_path)
        self.loader.load()
        self.results_dir = Path("results/multi_tier_evaluation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_all_scenarios(self):
        """اجرای تمام سناریوها"""
        print("\n" + "="*60)
        print("🚀 شروع اجرای آزمایش‌های چندسطحی")
        print("="*60)
        
        all_results = []
        
        for idx, scenario in enumerate(self.loader.scenarios, 1):
            print(f"\n📍 [{idx}/{len(self.loader.scenarios)}] اجرای سناریو: {scenario.id}")
            print(f"   Tier: {scenario.tier} | Complexity: {scenario.complexity}")
            
            result = self._run_single_scenario(scenario)
            all_results.append(result)
            
            # ذخیره نتایج جداگانه
            self._save_scenario_result(scenario, result)
        
        # ذخیره نتایج کلی
        self._save_final_results(all_results)
        
        print("\n" + "="*60)
        print("✅ تمام آزمایش‌ها با موفقیت انجام شد")
        print(f"📁 نتایج در: {self.results_dir}")
        print("="*60)
        
    def _run_single_scenario(self, scenario: Scenario) -> Dict:
        """اجرای یک سناریو و استخراج نتایج"""
        
        # شبیه‌سازی آموزش (در اینجا داده‌های شبیه‌سازی شده استفاده می‌شود)
        # در پیاده‌سازی واقعی، این قسمت با MADDPG جایگزین می‌شود
        num_episodes = 4000
        simulated_rewards = self._simulate_training(scenario, num_episodes)
        
        # محاسبه متریک‌های عملکرد
        metrics = self._calculate_metrics(scenario, simulated_rewards)
        
        result = {
            "scenario_id": scenario.id,
            "tier": scenario.tier,
            "complexity": scenario.complexity,
            "config": {
                "num_tasks": scenario.complexity_specs.num_tasks,
                "num_uavs": scenario.complexity_specs.num_uavs,
                "processing_capacity": scenario.tier_specs.processing_capacity,
                "communication_delay": scenario.tier_specs.communication_delay,
                "energy_per_flop": scenario.tier_specs.energy_per_flop,
                "reliability": scenario.tier_specs.reliability
            },
            "training_results": {
                "total_episodes": num_episodes,
                "final_reward": simulated_rewards[-1],
                "avg_reward_last_100": np.mean(simulated_rewards[-100:]),
                "convergence_episode": self._find_convergence(simulated_rewards),
                "reward_history": simulated_rewards.tolist()
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _simulate_training(self, scenario: Scenario, num_episodes: int) -> np.ndarray:
        """شبیه‌سازی فرآیند آموزش (جایگزین موقت MADDPG)"""
        
        # پارامترهای شبیه‌سازی بر اساس Tier و Complexity
        tier_factors = {
            "ground": 0.6,
            "edge": 0.75,
            "fog": 0.85,
            "cloud": 0.95
        }
        
        complexity_factors = {
            "easy": 0.9,
            "medium": 0.75,
            "hard": 0.6
        }
        
        base_reward = -100
        improvement_rate = 0.002
        tier_factor = tier_factors[scenario.tier]
        complexity_factor = complexity_factors[scenario.complexity]
        
        rewards = []
        for ep in range(num_episodes):
            progress = min(1.0, ep / (num_episodes * 0.7))
            reward = base_reward * (1 - progress * tier_factor * complexity_factor)
            noise = np.random.normal(0, 5)
            rewards.append(reward + noise)
        
        return np.array(rewards)
    
    def _find_convergence(self, rewards: np.ndarray, window: int = 100, threshold: float = 5.0) -> int:
        """تشخیص نقطه همگرایی"""
        if len(rewards) < window:
            return len(rewards)
        
        for i in range(window, len(rewards)):
            recent_std = np.std(rewards[i-window:i])
            if recent_std < threshold:
                return i
        
        return len(rewards)
    
    def _calculate_metrics(self, scenario: Scenario, rewards: np.ndarray) -> Dict:
        """محاسبه متریک‌های عملکردی"""
        
        specs = scenario.tier_specs
        complexity = scenario.complexity_specs
        
        # محاسبه Latency (ms)
        latency = (
            specs.communication_delay * 1000 +  # تبدیل به میلی‌ثانیه
            (complexity.num_tasks * 100) / specs.processing_capacity
        )
        
        # محاسبه Energy Consumption (Joule)
        avg_task_size = np.mean(complexity.task_size_range)  # MB
        flops_per_task = avg_task_size * 1e6  # تخمین FLOP
        energy = (
            complexity.num_tasks * flops_per_task * specs.energy_per_flop +
            specs.transmission_power * specs.communication_delay * complexity.num_uavs
        )
        
        # محاسبه Scalability Score (0-1)
        max_tasks = 50  # حداکثر تعداد Task در سناریوهای Hard
        scalability = 1 - (complexity.num_tasks / max_tasks) * (1 - specs.reliability)
        
        # محاسبه Success Rate
        success_rate = min(1.0, specs.reliability * (1 + np.mean(rewards[-100:]) / 100))
        
        # ⭐ محاسبه Throughput (tasks/sec)
        # فرمول: تعداد تسک‌ها / (زمان پردازش + تاخیر ارتباطی)
        processing_time = (complexity.num_tasks * 100) / specs.processing_capacity  # ثانیه
        total_time = processing_time + specs.communication_delay  # ثانیه
        throughput = complexity.num_tasks / max(total_time, 0.001)  # جلوگیری از تقسیم بر صفر
        
        return {
            "latency_ms": round(latency, 2),
            "energy_joules": round(energy, 4),
            "scalability_score": round(scalability, 4),
            "success_rate": round(max(0.0, success_rate), 4),
            "throughput": round(throughput, 2)  # ⭐ اضافه شد
        }
    
    def _save_scenario_result(self, scenario: Scenario, result: Dict):
        """ذخیره نتایج یک سناریو"""
        scenario_dir = self.results_dir / scenario.id
        scenario_dir.mkdir(exist_ok=True)
        
        output_file = scenario_dir / "result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 ذخیره شد: {output_file}")
    
    def _save_final_results(self, all_results: List[Dict]):
        """ذخیره نتایج نهایی کلی"""
        output_file = self.results_dir / "final_results.json"
        
        summary = {
            "metadata": {
                "total_scenarios": len(all_results),
                "timestamp": datetime.now().isoformat(),
                "description": "Multi-Tier Evaluation Results"
            },
            "results": all_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 نتایج نهایی ذخیره شد: {output_file}")


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_all_scenarios()
