"""
مقایسه چند مدل با یکدیگر
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from .model_evaluator import ModelEvaluator


class ModelComparison:
    """مقایسه چند مدل آموزش‌دیده"""
    
    def __init__(self, model_paths: List[str]):
        """
        Args:
            model_paths: لیست مسیرهای مدل‌ها
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.evaluators = []
        self.model_names = []
        
        for path in self.model_paths:
            try:
                evaluator = ModelEvaluator(str(path))
                self.evaluators.append(evaluator)
                # نام مدل از نام پوشه
                self.model_names.append(path.parent.name)
            except Exception as e:
                print(f"⚠️  Could not load model from {path}: {e}")
    
    def compare(self, num_episodes: int = 50) -> Dict:
        """
        مقایسه مدل‌ها
        
        Args:
            num_episodes: تعداد اپیزودها برای هر مدل
        
        Returns:
            دیکشنری حاوی نتایج مقایسه
        """
        if not self.evaluators:
            return {'error': 'No valid models loaded'}
        
        print(f"\n{'='*70}")
        print(f"🔍 Comparing {len(self.evaluators)} Models")
        print(f"{'='*70}\n")
        
        all_results = {}
        
        for i, (evaluator, name) in enumerate(zip(self.evaluators, self.model_names)):
            print(f"📊 Evaluating Model {i+1}/{len(self.evaluators)}: {name}")
            
            results = evaluator.evaluate(num_episodes=num_episodes, detailed=False)
            all_results[name] = results
            
            print(f"   Mean Reward: {results['statistics']['mean_reward']:.2f}\n")
        
        # تحلیل مقایسه‌ای
        comparison = self._analyze_comparison(all_results)
        
        # نمایش نتایج
        self._print_comparison(comparison)
        
        return {
            'models': all_results,
            'comparison': comparison
        }
    
    def _analyze_comparison(self, results: Dict) -> Dict:
        """تحلیل مقایسه‌ای مدل‌ها"""
        means = {name: res['statistics']['mean_reward'] 
                for name, res in results.items()}
        stds = {name: res['statistics']['std_reward'] 
               for name, res in results.items()}
        
        # پیدا کردن بهترین و بدترین
        best_model = max(means.items(), key=lambda x: x[1])
        worst_model = min(means.items(), key=lambda x: x[1])
        
        # محاسبه رتبه‌بندی
        ranked = sorted(means.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'best_model': {
                'name': best_model[0],
                'mean_reward': float(best_model[1]),
                'std_reward': float(stds[best_model[0]])
            },
            'worst_model': {
                'name': worst_model[0],
                'mean_reward': float(worst_model[1]),
                'std_reward': float(stds[worst_model[0]])
            },
            'ranking': [
                {
                    'rank': i + 1,
                    'name': name,
                    'mean_reward': float(reward),
                    'std_reward': float(stds[name])
                }
                for i, (name, reward) in enumerate(ranked)
            ],
            'performance_gap': float(best_model[1] - worst_model[1])
        }
    
    def _print_comparison(self, comparison: Dict):
        """چاپ نتایج مقایسه"""
        print(f"\n{'='*70}")
        print(f"🏆 Comparison Results")
        print(f"{'='*70}\n")
        
        print(f"📊 Ranking:")
        for rank_info in comparison['ranking']:
            print(f"   {rank_info['rank']}. {rank_info['name']}")
            print(f"      Mean: {rank_info['mean_reward']:.2f} ± {rank_info['std_reward']:.2f}")
        
        print(f"\n✨ Best Model: {comparison['best_model']['name']}")
        print(f"   Reward: {comparison['best_model']['mean_reward']:.2f}")
        
        print(f"\n⚠️  Performance Gap: {comparison['performance_gap']:.2f}\n")
