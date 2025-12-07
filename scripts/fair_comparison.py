#!/usr/bin/env python3
"""
Fair comparison between MADDPG and I-DDPG on the same environment
"""
import subprocess
import json
import os
from pathlib import Path

def run_experiment(algorithm, env_type, episodes=500):
    """Run training for a specific algorithm"""
    
    print(f"\n{'='*70}")
    print(f"🚀 Running {algorithm} on {env_type} environment")
    print(f"{'='*70}\n")
    
    if algorithm == "MADDPG":
        script = "scripts/train_maddpg_organized_v2.py"
    elif algorithm == "I-DDPG":
        script = "scripts/train_iddpg.py"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Run training
    cmd = ["python", script, "--env", env_type, "--episodes", str(episodes)]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {algorithm}:")
        print(e.stderr)
        return False

def main():
    """Run fair comparison"""
    
    env_type = "fake"  # یا "real" برای محیط واقعی
    episodes = 500
    
    results = {}
    
    # Run I-DDPG
    if run_experiment("I-DDPG", env_type, episodes):
        results["I-DDPG"] = "Success"
    else:
        results["I-DDPG"] = "Failed"
    
    # Run MADDPG
    if run_experiment("MADDPG", env_type, episodes):
        results["MADDPG"] = "Success"
    else:
        results["MADDPG"] = "Failed"
    
    # Save results
    output_dir = Path("results/fair_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("📊 COMPARISON COMPLETE")
    print("="*70)
    print(json.dumps(results, indent=2))
    print("\nResults saved to: results/fair_comparison/")

if __name__ == "__main__":
    main()
