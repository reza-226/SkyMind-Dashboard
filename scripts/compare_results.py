"""
Compare results between MADDPG and baseline algorithms.
Generates comparison plots and statistical analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(result_dir):
    """Load results from JSON file"""
    result_path = Path(result_dir) / 'results.json'
    if not result_path.exists():
        print(f"⚠️  Results not found: {result_path}")
        return None
    
    with open(result_path, 'r') as f:
        return json.load(f)


def compare_algorithms():
    """
    Compare MADDPG with all baselines.
    """
    algorithms = ['maddpg', 'random', 'ddpg', 'ppo', 'iddpg']
    results = {}
    
    print("\n" + "="*60)
    print("📊 Loading Results for Comparison")
    print("="*60 + "\n")
    
    # Load MADDPG results
    maddpg_results = load_results('results/maddpg')
    if maddpg_results:
        results['maddpg'] = maddpg_results
        print(f"✅ MADDPG: Mean Reward = {maddpg_results.get('mean_reward', 'N/A')}")
    
    # Load baseline results
    for algo in ['random', 'ddpg', 'ppo', 'iddpg']:
        baseline_results = load_results(f'results/baselines/{algo}')
        if baseline_results:
            results[algo] = baseline_results
            print(f"✅ {algo.upper()}: Mean Reward = {baseline_results.get('mean_reward', 'N/A')}")
        else:
            print(f"❌ {algo.upper()}: Results not available")
    
    if len(results) < 2:
        print("\n⚠️  Need at least 2 algorithms to compare!")
        return
    
    # TODO: Generate comparison plots
    print("\n📈 Comparison analysis ready for implementation")
    print("   Features to add:")
    print("   - Learning curves comparison")
    print("   - Box plots of rewards")
    print("   - Statistical significance tests")
    print("   - Performance tables")


if __name__ == '__main__':
    compare_algorithms()
