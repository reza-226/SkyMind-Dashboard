"""
Main script to run baseline algorithms for comparison with MADDPG.
Usage:
    python scripts/run_baseline.py --algorithm random --episodes 500
    python scripts/run_baseline.py --algorithm ddpg --episodes 1000
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.baselines.random_policy import RandomAgent, run_random_baseline


def main():
    parser = argparse.ArgumentParser(description='Run baseline algorithms')
    parser.add_argument('--algorithm', type=str, required=True,
                        choices=['random', 'ddpg', 'ppo', 'iddpg'],
                        help='Baseline algorithm to run')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Set save directory
    if args.save_dir is None:
        args.save_dir = f'results/baselines/{args.algorithm}'
    
    print(f"\n{'='*70}")
    print(f"🚀 Running Baseline: {args.algorithm.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Save Directory: {args.save_dir}")
    print(f"{'='*70}\n")
    
    # TODO: Load environment from config
    # For now, this is a placeholder
    env = None  # Replace with actual environment initialization
    
    if args.algorithm == 'random':
        if env is None:
            print("⚠️  Environment not initialized - skipping execution")
            print("📝 Random policy implementation is ready!")
            print("   Next step: Initialize environment and run")
            return
        
        results = run_random_baseline(
            env=env,
            num_episodes=args.episodes,
            save_dir=args.save_dir
        )
        
    elif args.algorithm == 'ddpg':
        print("❌ DDPG implementation pending")
        print("   Use --algorithm random for now")
        
    elif args.algorithm == 'ppo':
        print("❌ PPO implementation pending")
        print("   Use --algorithm random for now")
        
    elif args.algorithm == 'iddpg':
        print("❌ Independent DDPG implementation pending")
        print("   Use --algorithm random for now")
    
    print("\n✅ Baseline execution completed!")


if __name__ == '__main__':
    main()
