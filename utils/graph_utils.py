"""
ابزارهای کار با گراف‌های DAG برای پروژه SkyMind
"""
import os
import json
import random
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    """وضعیت‌های ممکن برای یک task"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskNode:
    """
    نمایش یک task در DAG
    
    Attributes:
        task_id: شناسه یکتای task
        data_size: حجم داده ورودی (MB)
        comp_requirement: نیاز محاسباتی (CPU cycles)
        deadline: مهلت زمانی (seconds)
        priority: اولویت task (1-10)
        status: وضعیت فعلی task
        dependencies: لیست task_id های والد
        successors: لیست task_id های فرزند
    """
    task_id: int
    data_size: float
    comp_requirement: float
    deadline: float
    priority: float = 5.0
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[int] = None
    successors: List[int] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.successors is None:
            self.successors = []
    
    def __hash__(self):
        return hash(self.task_id)


class TaskDAG:
    """
    کلاس DAG برای مدیریت وابستگی‌های task
    """
    
    def __init__(self, dag_id: str):
        """
        Args:
            dag_id: شناسه یکتای DAG
        """
        self.dag_id = dag_id
        self.nodes: Dict[int, TaskNode] = {}
        self._ready_cache: Optional[Set[int]] = None
    
    def add_task(self, task: TaskNode):
        """
        اضافه کردن task به DAG
        
        Args:
            task: شیء TaskNode
        """
        if task.task_id in self.nodes:
            raise ValueError(f"Task {task.task_id} already exists in DAG")
        
        self.nodes[task.task_id] = task
        self._ready_cache = None  # invalidate cache
    
    def add_dependency(self, parent_id: int, child_id: int):
        """
        اضافه کردن وابستگی بین دو task
        
        Args:
            parent_id: task والد (باید قبل از child اجرا شود)
            child_id: task فرزند
        """
        if parent_id not in self.nodes:
            raise ValueError(f"Parent task {parent_id} does not exist")
        if child_id not in self.nodes:
            raise ValueError(f"Child task {child_id} does not exist")
        
        # جلوگیری از ایجاد حلقه
        if self._would_create_cycle(parent_id, child_id):
            raise ValueError(f"Adding edge {parent_id}->{child_id} would create a cycle")
        
        # اضافه کردن رابطه
        if child_id not in self.nodes[parent_id].successors:
            self.nodes[parent_id].successors.append(child_id)
        
        if parent_id not in self.nodes[child_id].dependencies:
            self.nodes[child_id].dependencies.append(parent_id)
        
        self._ready_cache = None
    
    def _would_create_cycle(self, parent_id: int, child_id: int) -> bool:
        """بررسی اینکه آیا اضافه کردن یال حلقه ایجاد می‌کند"""
        # DFS از child به parent
        visited = set()
        stack = [child_id]
        
        while stack:
            current = stack.pop()
            if current == parent_id:
                return True
            
            if current in visited:
                continue
            
            visited.add(current)
            stack.extend(self.nodes[current].successors)
        
        return False
    
    def get_ready_tasks(self) -> Set[int]:
        """
        بازگرداندن task‌هایی که آماده اجرا هستند
        (تمام dependencies آن‌ها complete شده‌اند)
        """
        if self._ready_cache is not None:
            return self._ready_cache
        
        ready = set()
        for task_id, task in self.nodes.items():
            if task.status == TaskStatus.PENDING:
                # بررسی کامل بودن تمام dependencies
                all_deps_complete = all(
                    self.nodes[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if all_deps_complete:
                    ready.add(task_id)
        
        self._ready_cache = ready
        return ready
    
    def mark_completed(self, task_id: int):
        """علامت‌گذاری task به عنوان complete"""
        if task_id not in self.nodes:
            raise ValueError(f"Task {task_id} does not exist")
        
        self.nodes[task_id].status = TaskStatus.COMPLETED
        self._ready_cache = None
    
    def __len__(self):
        """تعداد task‌ها"""
        return len(self.nodes)
    
    def __repr__(self):
        return f"TaskDAG(dag_id={self.dag_id}, num_tasks={len(self.nodes)})"


def convert_dag_to_pyg_data(dag: TaskDAG) -> Data:
    """
    تبدیل TaskDAG به PyTorch Geometric Data object

    Args:
        dag: شیء TaskDAG

    Returns:
        data: PyTorch Geometric Data object حاوی:
            - x: node features [num_nodes, 9]
            - edge_index: edge indices [2, num_edges]
            - edge_attr: edge features [num_edges, 3]
    """
    # استخراج node features
    node_features = []
    task_ids = sorted(dag.nodes.keys())
    
    # محاسبه max برای نرمال‌سازی
    max_comp = max(dag.nodes[tid].comp_requirement for tid in task_ids) if task_ids else 1.0
    max_data = max(dag.nodes[tid].data_size for tid in task_ids) if task_ids else 1.0
    
    for task_id in task_ids:
        task = dag.nodes[task_id]
        
        # ویژگی‌های task:
        # [comp_requirement, data_size, deadline, in_degree, out_degree,
        #  is_ready, is_completed, priority, normalized_comp]
        
        in_degree = len(task.dependencies)
        out_degree = len(task.successors)
        
        # بررسی آماده بودن task
        ready_tasks = dag.get_ready_tasks()
        is_ready = 1.0 if task_id in ready_tasks else 0.0
        
        # بررسی تکمیل task
        is_completed = 1.0 if task.status == TaskStatus.COMPLETED else 0.0
        
        # نرمال‌سازی
        normalized_comp = task.comp_requirement / max_comp if max_comp > 0 else 0.0
        normalized_data = task.data_size / max_data if max_data > 0 else 0.0
        
        features = [
            float(task.comp_requirement),
            float(task.data_size),
            float(task.deadline),
            float(in_degree),
            float(out_degree),
            is_ready,
            is_completed,
            float(task.priority),
            normalized_comp
        ]
        node_features.append(features)
    
    # تبدیل به tensor
    x = torch.tensor(node_features, dtype=torch.float32)
    
    # استخراج edge_index و edge_attr
    edge_list = []
    edge_features = []
    
    for task_id in task_ids:
        task = dag.nodes[task_id]
        for successor_id in task.successors:
            # یال از task_id به successor_id
            src_idx = task_ids.index(task_id)
            dst_idx = task_ids.index(successor_id)
            edge_list.append([src_idx, dst_idx])
            
            # ویژگی یال: [data_transfer_size, is_critical, weight]
            data_transfer = task.data_size
            is_critical = 0.0  # می‌تواند با الگوریتم Critical Path محاسبه شود
            weight = 1.0
            
            edge_features.append([data_transfer, is_critical, weight])
    
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
    
    # ساخت PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    return data


def calculate_critical_path(dag: TaskDAG) -> Tuple[List[int], float]:
    """
    محاسبه مسیر بحرانی (Critical Path) در DAG
    
    Args:
        dag: شیء TaskDAG
        
    Returns:
        critical_path: لیست task_id های موجود در مسیر بحرانی
        critical_path_length: طول مسیر بحرانی
    """
    G = nx.DiGraph()
    
    for task_id, task in dag.nodes.items():
        G.add_node(task_id, weight=task.comp_requirement)
        
    for task_id, task in dag.nodes.items():
        for successor_id in task.successors:
            G.add_edge(task_id, successor_id)
    
    try:
        critical_path = nx.dag_longest_path(G, weight='weight')
        critical_path_length = nx.dag_longest_path_length(G, weight='weight')
    except:
        critical_path = []
        critical_path_length = 0.0
        
    return critical_path, critical_path_length


def generate_random_dag(
    num_tasks: int = 10,
    edge_probability: float = 0.3,
    seed: Optional[int] = None
) -> TaskDAG:
    """
    تولید یک DAG تصادفی برای تست
    
    Args:
        num_tasks: تعداد tasks
        edge_probability: احتمال وجود یال بین دو task
        seed: random seed
        
    Returns:
        dag: TaskDAG تصادفی
    """
    import random
    if seed is not None:
        random.seed(seed)
    
    dag = TaskDAG(dag_id=f"random_dag_{seed}")
    
    # ایجاد tasks
    for i in range(num_tasks):
        task = TaskNode(
            task_id=i,
            data_size=random.uniform(1.0, 10.0),
            comp_requirement=random.uniform(500, 2000),
            deadline=random.uniform(5, 20),
            priority=random.uniform(1, 10)
        )
        dag.add_task(task)
    
    # اضافه کردن dependencies (فقط از i به j که j > i)
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if random.random() < edge_probability:
                try:
                    dag.add_dependency(i, j)
                except ValueError:
                    # حلقه ایجاد می‌شود، skip
                    pass
    
    return dag


def get_dag_statistics(dag: TaskDAG) -> Dict:
    """
    محاسبه آمار کلی DAG
    
    Args:
        dag: شیء TaskDAG
        
    Returns:
        stats: دیکشنری حاوی آمار DAG
    """
    task_ids = list(dag.nodes.keys())
    
    in_degrees = [len(dag.nodes[tid].dependencies) for tid in task_ids]
    out_degrees = [len(dag.nodes[tid].successors) for tid in task_ids]
    
    critical_path, critical_length = calculate_critical_path(dag)
    
    stats = {
        'num_nodes': len(task_ids),
        'num_edges': sum(out_degrees),
        'avg_in_degree': sum(in_degrees) / len(in_degrees) if in_degrees else 0,
        'avg_out_degree': sum(out_degrees) / len(out_degrees) if out_degrees else 0,
        'max_in_degree': max(in_degrees) if in_degrees else 0,
        'max_out_degree': max(out_degrees) if out_degrees else 0,
        'critical_path_length': critical_length,
        'critical_path_nodes': len(critical_path)
    }
    
    return stats
def load_random_dag(stage: str = "easy"):
    """
    Load a random DAG from datasets/dags/<stage>/*.json
    If no JSON exists, generate a random DAG.

    Args:
        stage: 'easy', 'medium', 'hard'

    Returns:
        pyg_data: PyTorch Geometric Data object
    """
    base_dir = os.path.join("datasets", "dags", stage)

    # اگر پوشه وجود دارد و فایل JSON در آن هست
    if os.path.exists(base_dir):
        files = [f for f in os.listdir(base_dir) if f.endswith(".json")]

        if len(files) > 0:
            selected = random.choice(files)
            path = os.path.join(base_dir, selected)

            with open(path, "r") as f:
                dag_dict = json.load(f)

            # ساخت DAG دستی از JSON
            dag = TaskDAG(dag_id=dag_dict.get("dag_id", selected))

            # 1) ساخت تمام تسک‌ها
            for node in dag_dict["nodes"]:
                t = TaskNode(
                    task_id=node["task_id"],
                    data_size=node["data_size"],
                    comp_requirement=node["comp_requirement"],
                    deadline=node["deadline"],
                    priority=node.get("priority", 5.0)
                )
                dag.add_task(t)

            # 2) اضافه کردن وابستگی‌ها
            for parent, child in dag_dict["edges"]:
                dag.add_dependency(parent, child)

            return convert_dag_to_pyg_data(dag)

    # ---- اگر JSON نبود → fallback: تولید DAG تصادفی ----
    random_dag = generate_random_dag(
        num_tasks=10,
        edge_probability=0.3,
        seed=random.randint(0, 99999)
    )
    return convert_dag_to_pyg_data(random_dag)
def get_root_node(dag):
    """
    Find root node(s) in DAG (nodes with no incoming edges)
    Args:
        dag: torch_geometric.data.Data with edge_index
    Returns:
        int: index of first root node
    """
    edge_index = dag.edge_index
    num_nodes = dag.num_nodes
    
    # nodes with incoming edges
    target_nodes = set(edge_index[1].tolist())
    
    # find nodes WITHOUT incoming edges (roots)
    roots = [i for i in range(num_nodes) if i not in target_nodes]
    
    if len(roots) == 0:
        # fallback: return first node
        return 0
    
    return roots[0]
