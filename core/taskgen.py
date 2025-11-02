# core/taskgen.py
import numpy as np

class TaskGenerator:
    def __init__(self, num_tasks=10):
        self.num_tasks = num_tasks

    def generate_task_dag(self):
        """Generate a simple DAG for dependent tasks."""
        G = {i: [] for i in range(self.num_tasks)}
        for i in range(self.num_tasks):
            for j in range(i+1, self.num_tasks):
                if np.random.rand() < 0.2:
                    G[i].append(j)
        return G

    def generate_task_profiles(self):
        """Each task has size (MB) and CPU cycles (MHz)."""
        sizes = np.random.exponential(5, self.num_tasks)
        cycles = np.random.uniform(500, 2000, self.num_tasks)
        return [{"size": s, "cycles": c} for s, c in zip(sizes, cycles)]
