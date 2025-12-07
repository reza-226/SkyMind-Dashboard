"""
تحلیل تاریخچه آموزش
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class TrainingAnalyzer:
    """تحلیل فایل‌های تاریخچه آموزش"""
    
    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: مسیر پوشه نتایج آموزش
        """
        self.results_dir = Path(results_dir)
        self.training_file = self.results_dir / 'training_results.json'
        
        if not self.training_file.exists():
            print(f"⚠️  Warning: Training file not found: {self.training_file}")
    
    def analyze(self) -> Dict:
        """تحلیل تاریخچه آموزش"""
        
        if not self.training_file.exists():
            return {
                'error': 'Training results file not found',
                'path': str(self.training_file)
            }
        
        # بارگذاری داده‌های آموزش
        with open(self.training_file, 'r') as f:
            training_data = json.load(f)
        
        episodes = training_data.get('episodes', [])
        
        if not episodes:
            return {
                'error': 'No episode data found in training results'
            }
        
        # استخراج اطلاعات
        episode_numbers = [ep['episode'] for ep in episodes]
        rewards = [ep['reward'] for ep in episodes]
        
        # محاسبه آمار
        results = {
            'total_episodes': len(episodes),
            'final_reward': float(rewards[-1]),
            'best_reward': float(max(rewards)),
            'worst_reward': float(min(rewards)),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'reward_trend': self._calculate_trend(rewards),
            'convergence_episode': self._find_convergence(rewards),
            'episodes': episode_numbers,
            'rewards': rewards
        }
        
        # تحلیل بخش‌های مختلف آموزش
        results['phases'] = self._analyze_phases(rewards)
        
        return results
    
    def _calculate_trend(self, rewards: List[float], window: int = 100) -> str:
        """محاسبه روند کلی reward"""
        if len(rewards) < window:
            return "insufficient_data"
        
        first_half = np.mean(rewards[:len(rewards)//2])
        second_half = np.mean(rewards[len(rewards)//2:])
        
        improvement = ((second_half - first_half) / abs(first_half)) * 100
        
        if improvement > 10:
            return "improving"
        elif improvement < -10:
            return "declining"
        else:
            return "stable"
    
    def _find_convergence(self, rewards: List[float], window: int = 50, threshold: float = 0.05) -> Optional[int]:
        """پیدا کردن نقطه همگرایی (تقریبی)"""
        if len(rewards) < window * 2:
            return None
        
        for i in range(window, len(rewards) - window):
            prev_mean = np.mean(rewards[i-window:i])
            next_mean = np.mean(rewards[i:i+window])
            
            if abs(next_mean - prev_mean) / abs(prev_mean) < threshold:
                return i
        
        return None
    
    def _analyze_phases(self, rewards: List[float]) -> Dict:
        """تحلیل فازهای مختلف آموزش"""
        total = len(rewards)
        
        phase_size = total // 4
        
        phases = {
            'early': {
                'range': (0, phase_size),
                'mean': float(np.mean(rewards[:phase_size])),
                'std': float(np.std(rewards[:phase_size]))
            },
            'mid': {
                'range': (phase_size, phase_size*3),
                'mean': float(np.mean(rewards[phase_size:phase_size*3])),
                'std': float(np.std(rewards[phase_size:phase_size*3]))
            },
            'late': {
                'range': (phase_size*3, total),
                'mean': float(np.mean(rewards[phase_size*3:])),
                'std': float(np.std(rewards[phase_size*3:]))
            }
        }
        
        return phases
