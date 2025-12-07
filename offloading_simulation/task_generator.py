"""
تولید Task با سطوح پیچیدگی مختلف
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List

class TaskComplexity(Enum):
    """سطوح پیچیدگی Task"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

@dataclass
class Task:
    """کلاس Task"""
    task_id: int
    complexity: TaskComplexity
    computational_load: float  # GFLOPS
    data_size: float  # MB
    deadline: float  # ms
    priority: int  # 1-10
    
    def __repr__(self):
        return f"Task({self.task_id}, {self.complexity.value}, {self.computational_load:.2f}G, {self.data_size:.2f}MB)"


class TaskGenerator:
    """تولیدکننده Task"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # پارامترهای توزیع برای هر سطح پیچیدگی
        self.complexity_params = {
            TaskComplexity.SIMPLE: {
                "comp_load": (0.5, 2.0),  # GFLOPS (min, max)
                "data_size": (0.1, 1.0),  # MB
                "deadline": (50, 200),    # ms
                "priority": (1, 3)
            },
            TaskComplexity.MEDIUM: {
                "comp_load": (2.0, 10.0),
                "data_size": (1.0, 5.0),
                "deadline": (200, 500),
                "priority": (4, 7)
            },
            TaskComplexity.COMPLEX: {
                "comp_load": (10.0, 50.0),
                "data_size": (5.0, 20.0),
                "deadline": (500, 2000),
                "priority": (7, 10)
            }
        }
    
    def generate_task(self, task_id: int, complexity: TaskComplexity) -> Task:
        """تولید یک Task"""
        params = self.complexity_params[complexity]
        
        return Task(
            task_id=task_id,
            complexity=complexity,
            computational_load=np.random.uniform(*params["comp_load"]),
            data_size=np.random.uniform(*params["data_size"]),
            deadline=np.random.uniform(*params["deadline"]),
            priority=np.random.randint(*params["priority"])
        )
    
    def generate_batch(self, 
                      num_tasks: int,
                      complexity_distribution: dict = None) -> List[Task]:
        """
        تولید دسته‌ای از Task‌ها
        
        Args:
            num_tasks: تعداد Task
            complexity_distribution: توزیع پیچیدگی (مثلاً {"simple": 0.5, "medium": 0.3, "complex": 0.2})
        """
        if complexity_distribution is None:
            # توزیع پیش‌فرض
            complexity_distribution = {
                "simple": 0.5,
                "medium": 0.3,
                "complex": 0.2
            }
        
        tasks = []
        complexities = list(TaskComplexity)
        probs = [complexity_distribution.get(c.value, 0) for c in complexities]
        
        for i in range(num_tasks):
            complexity = np.random.choice(complexities, p=probs)
            task = self.generate_task(i, complexity)
            tasks.append(task)
        
        return tasks
