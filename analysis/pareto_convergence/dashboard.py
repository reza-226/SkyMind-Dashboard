# D:\Payannameh\SkyMind-Dashboard\analysis\pareto_convergence\dashboard.py
import numpy as np

class NSGAII:
    """Pareto-based multi-objective optimization integrated with MADDPG and DTLCM"""

    def __init__(self, population_size=50, generations=100, fitness_fn=None):
        self.population_size = population_size
        self.generations = generations
        self.fitness_fn = fitness_fn or self.default_fitness
        self.archive = []

    def default_fitness(self, x):
        # Placeholder for trade-off between energy and latency (C3 category)
        e, l = x
        return e**2 + l**2

    def optimize(self):
        pop = np.random.rand(self.population_size, 2)
        for g in range(self.generations):
            fitness = np.array([self.fitness_fn(ind) for ind in pop])
            # Simple non-dominated sorting imitation
            ranks = np.argsort(fitness)
            pop = pop[ranks[:self.population_size]]
            self.archive.append(pop.mean(axis=0))
        return self.archive[-1]

    def to_json(self, path="analysis/realtime/pareto_snapshot.json"):
        import json, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        best = self.archive[-1] if self.archive else [0, 0]
        with open(path, "w") as f:
            json.dump({"pareto_best": list(best)}, f)
