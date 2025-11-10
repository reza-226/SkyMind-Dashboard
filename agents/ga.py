# agents/ga.py
class NSGAII:
    def __init__(self, pop_size=None, generations=None, objectives=None,
                 predation_model='Levy', mutation_rate=0.05):
        # backward-compatible support
        self.pop_size = pop_size or 50
        self.generations = generations or 10
        self.objectives = objectives or ['energy','latency']
        self.predation_model = predation_model
        self.mutation_rate = mutation_rate
        print(f"[Ninja] 🧩 NSGA-II/MP Optimizer (pop={self.pop_size}, gen={self.generations}, model={self.predation_model}) initialized.")
