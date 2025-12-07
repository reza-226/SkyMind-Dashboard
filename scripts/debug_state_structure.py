#!/usr/bin/env python3
"""
Debug script: Verify state structure and info metrics
Confirms that the 4-layer metrics (Delay, Energy, Distance, QoS) exist in info dict
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.env.environment import UAVMECEnvironment

def create_dummy_dag(num_nodes=5):
    """
    Create a DAG dictionary matching the environment's expectations.
    Based on environment.py line 42: dag is a dictionary
    """
    # Create random node features (8D to match GNN)
    node_features = torch.randn(num_nodes, 8)
    
    # Create simple chain edges
    edges = [(i, i+1) for i in range(num_nodes-1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Edge features (3D)
    edge_features = torch.randn(edge_index.size(1), 3)
    
    # Return as dictionary (not PyG Data object)
    return {
        'num_nodes': num_nodes,
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': edge_features
    }

def main():
    print("=" * 80)
    print("STATE STRUCTURE & INFO METRICS DEBUG")
    print("=" * 80)
    
    # Initialize environment with correct signature
    print("\n[1] Initializing UAVMECEnvironment...")
    print("    (device='cpu', max_steps=100)")
    env = UAVMECEnvironment(device="cpu", max_steps=100)
    
    # Reset environment
    print("\n[2] Resetting environment...")
    dummy_dag = create_dummy_dag(num_nodes=5)
    print(f"    DAG nodes: {dummy_dag['num_nodes']}")
    print(f"    Node features shape: {dummy_dag['node_features'].shape}")
    
    state = env.reset(dag=dummy_dag)
    
    # Analyze state structure
    print("\n[3] STATE STRUCTURE:")
    print(f"    State type: {type(state)}")
    print(f"    State dtype: {state.dtype}")
    print(f"    State shape: {state.shape}")
    
    if len(state.shape) == 1:
        dims = state.shape[0]
    else:
        dims = state.shape[-1]
    
    print(f"    State dimensions: {dims}")
    
    if dims == 537:
        print("    ✅ State has 537 dimensions (256 graph + 256 node + 25 flat)")
    else:
        print(f"    ⚠️  Unexpected state dimension: {dims}")
        print(f"    Expected: 537, Got: {dims}")
    
    # Take a random step
    print("\n[4] Taking a random step...")
    action = {
        'offload': np.random.randint(0, 5),
        'cpu': np.random.uniform(0, 1),
        'bandwidth': np.random.dirichlet([1, 1, 1]),
        'move': np.random.uniform(-5, 5, size=2)
    }
    
    print(f"    Action keys: {list(action.keys())}")
    print(f"    - offload: {action['offload']}")
    print(f"    - cpu: {action['cpu']:.4f}")
    print(f"    - bandwidth: {action['bandwidth']}")
    print(f"    - move: {action['move']}")
    
    # Step returns: next_state, reward, terminated, truncated, info
    result = env.step(action)
    
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        has_info = True
    elif len(result) == 4:
        next_state, reward, terminated, info = result
        truncated = False
        has_info = True
    else:
        next_state, reward, terminated = result[:3]
        has_info = False
        info = {}
    
    print(f"\n[5] STEP RESULTS:")
    print(f"    Reward: {reward:.4f}")
    print(f"    Terminated: {terminated}")
    if len(result) == 5:
        print(f"    Truncated: {truncated}")
    print(f"    Has info dict: {has_info}")
    
    # Check for 4-layer metrics
    expected_metrics = ['delay', 'energy_consumption', 'distance', 'qos_satisfaction']
    
    if has_info and info:
        print("\n[6] INFO DICTIONARY CONTENT:")
        print(f"    Keys found: {list(info.keys())}")
        
        print("\n[7] CHECKING 4-LAYER METRICS:")
        for metric in expected_metrics:
            if metric in info:
                value = info[metric]
                print(f"    ✅ '{metric}': {value:.6f}")
            else:
                print(f"    ❌ '{metric}': NOT FOUND")
        
        # Show all info content
        print("\n[8] ALL INFO CONTENT:")
        for key in sorted(info.keys()):
            value = info[key]
            if isinstance(value, (int, float, np.number)):
                print(f"    {key}: {value:.6f}")
            elif isinstance(value, np.ndarray):
                print(f"    {key}: array{value.shape}")
            else:
                print(f"    {key}: {type(value).__name__}")
    else:
        print("\n[6] ⚠️  NO INFO DICTIONARY RETURNED!")
        print("    Environment may not be returning 5-tuple (Gymnasium style)")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"    State dimension: {dims}")
    if has_info:
        print(f"    Info keys: {len(info)}")
        missing_metrics = [m for m in expected_metrics if m not in info]
        if missing_metrics:
            print(f"    ⚠️  Missing metrics: {missing_metrics}")
            print("\n    ⚡ ACTION REQUIRED:")
            print("       These metrics should be added to environment.step() return")
        else:
            print("    ✅ All 4-layer metrics present in info!")
    else:
        print("    ⚠️  No info dict returned (old Gym API?)")

if __name__ == "__main__":
    main()
