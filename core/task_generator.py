# core/task_generator.py یا core/taskgen.py
import numpy as np
from typing import List, Dict, Optional

class TaskGenerator:
    """Generate tasks for UAV-MEC environment"""
    
    def __init__(self, n_uavs: int = 3, map_size: float = 1000.0, 
                 num_tasks: int = 10, seed: Optional[int] = None):
        """
        Initialize task generator.
        
        Args:
            n_uavs: Number of UAVs in the environment
            map_size: Size of the operational area (meters)
            num_tasks: Number of tasks to generate per episode (default)
            seed: Random seed for reproducibility
        """
        self.n_uavs = n_uavs
        self.map_size = map_size
        self.num_tasks = num_tasks
        self.rng = np.random.RandomState(seed)
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            np.random.seed(seed)
        return [seed]
    
    def generate_task_dag(self, n_tasks: Optional[int] = None) -> Dict[int, List[int]]:
        """
        Generate a simple DAG for dependent tasks.
        
        Args:
            n_tasks: Number of tasks (uses self.num_tasks if None)
        """
        n = n_tasks if n_tasks is not None else self.num_tasks
        G = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if self.rng.rand() < 0.2:
                    G[i].append(j)
        return G
    
    def generate_task_profiles(self, n_tasks: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Generate task profiles with size and computational requirements.
        
        Args:
            n_tasks: Number of tasks (uses self.num_tasks if None)
        """
        n = n_tasks if n_tasks is not None else self.num_tasks
        sizes = self.rng.exponential(5, n)
        cycles = self.rng.uniform(500, 2000, n)
        deadlines = self.rng.uniform(5, 20, n)
        
        return [
            {
                "size": float(s),
                "cycles": float(c),
                "deadline": float(d)
            }
            for s, c, d in zip(sizes, cycles, deadlines)
        ]
    
    def generate_task_locations(self, n_tasks: Optional[int] = None) -> np.ndarray:
        """
        Generate random task locations within the map.
        
        Args:
            n_tasks: Number of tasks (uses self.num_tasks if None)
        """
        n = n_tasks if n_tasks is not None else self.num_tasks
        locations = self.rng.uniform(0, self.map_size, size=(n, 2))
        altitudes = np.zeros((n, 1))
        return np.hstack([locations, altitudes])
    
    def generate_tasks(self, n_tasks: Optional[int] = None) -> Dict:
        """
        Generate complete task set.
        
        Args:
            n_tasks: Number of tasks to generate (uses self.num_tasks if None)
            
        Returns:
            Dictionary containing:
                - profiles: List of task profiles (size, cycles, deadline)
                - dag: Task dependency graph
                - locations: 3D locations of tasks
        """
        n = n_tasks if n_tasks is not None else self.num_tasks
        
        return {
            'profiles': self.generate_task_profiles(n_tasks=n),
            'dag': self.generate_task_dag(n_tasks=n),
            'locations': self.generate_task_locations(n_tasks=n)
        }
    
    def reset(self, seed: Optional[int] = None):
        """Reset the generator with optional new seed"""
        return self.seed(seed)
