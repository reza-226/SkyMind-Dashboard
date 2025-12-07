"""
Performance benchmarks for MADDPG Agents V2
Measures speed and memory usage
"""

import torch
import numpy as np
import time
from agents import ActorNetwork, CriticNetwork, ActionDecoder

def benchmark_actor(batch_sizes=[1, 8, 32, 128], num_iterations=100):
    """Benchmark actor network performance"""
    print("\n" + "="*60)
    print("Actor Network Performance Benchmark")
    print("="*60 + "\n")
    
    actor = ActorNetwork().cuda() if torch.cuda.is_available() else ActorNetwork()
    device = next(actor.parameters()).device
    
    for batch_size in batch_sizes:
        state = torch.randn(batch_size, 537).to(device)
        
        # Warmup
        for _ in range(10):
            _ = actor(state)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            offload_logits, cont = actor(state)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        fps = (batch_size * num_iterations) / elapsed
        
        print(f"Batch size {batch_size:3d}: {fps:8.1f} samples/sec  ({elapsed:.3f}s total)")

def benchmark_critic(batch_sizes=[1, 8, 32, 128], num_iterations=100):
    """Benchmark critic network performance"""
    print("\n" + "="*60)
    print("Critic Network Performance Benchmark")
    print("="*60 + "\n")
    
    critic = CriticNetwork().cuda() if torch.cuda.is_available() else CriticNetwork()
    device = next(critic.parameters()).device
    
    for batch_size in batch_sizes:
        state = torch.randn(batch_size, 537).to(device)
        action = torch.randn(batch_size, 11).to(device)
        
        # Warmup
        for _ in range(10):
            _ = critic(state, action)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            q_value = critic(state, action)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        fps = (batch_size * num_iterations) / elapsed
        
        print(f"Batch size {batch_size:3d}: {fps:8.1f} samples/sec  ({elapsed:.3f}s total)")

def benchmark_decoder(batch_sizes=[1, 8, 32, 128], num_iterations=1000):
    """Benchmark action decoder performance"""
    print("\n" + "="*60)
    print("Action Decoder Performance Benchmark")
    print("="*60 + "\n")
    
    decoder = ActionDecoder()
    
    for batch_size in batch_sizes:
        offload_logits = torch.randn(batch_size, 5)
        cont = torch.randn(batch_size, 6)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            actions = decoder.decode(offload_logits, cont)
        
        elapsed = time.time() - start
        fps = (batch_size * num_iterations) / elapsed
        
        print(f"Batch size {batch_size:3d}: {fps:8.1f} samples/sec  ({elapsed:.3f}s total)")

def memory_usage():
    """Measure memory usage of networks"""
    print("\n" + "="*60)
    print("Memory Usage")
    print("="*60 + "\n")
    
    actor = ActorNetwork()
    critic = CriticNetwork()
    
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    
    actor_mem = sum(p.numel() * p.element_size() for p in actor.parameters()) / (1024**2)
    critic_mem = sum(p.numel() * p.element_size() for p in critic.parameters()) / (1024**2)
    
    print(f"Actor Network:")
    print(f"  Parameters: {actor_params:,}")
    print(f"  Memory: {actor_mem:.2f} MB\n")
    
    print(f"Critic Network:")
    print(f"  Parameters: {critic_params:,}")
    print(f"  Memory: {critic_mem:.2f} MB\n")
    
    print(f"Total:")
    print(f"  Parameters: {actor_params + critic_params:,}")
    print(f"  Memory: {actor_mem + critic_mem:.2f} MB")

if __name__ == '__main__':
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\nRunning benchmarks on: {device_name}\n")
    
    benchmark_actor()
    benchmark_critic()
    benchmark_decoder()
    memory_usage()
    
    print("\n" + "="*60)
    print("Benchmark Complete")
    print("="*60 + "\n")
