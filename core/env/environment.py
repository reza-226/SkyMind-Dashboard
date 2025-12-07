import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from models.gnn.task_encoder import GNNTaskEncoder as TaskEncoder


class UAVMECEnvironment:
    """
    محیط UAV-MEC با معماری 4 لایه Offloading:
        - Layer 0: Local (UAV processing)
        - Layer 1: Edge Server
        - Layer 2: Fog Server  
        - Layer 3: Cloud Server
    """
    
    def __init__(self, device="cpu", max_steps=100):
        self.device = device
        self.max_steps = max_steps
        self.current_step = 0
        
        # Task encoder - بدون device در constructor
        self.task_encoder = TaskEncoder()
        self.task_encoder.to(device)
        
        # UAV state
        self.uav_position = np.array([50.0, 50.0])
        self.uav_energy = 1.0
        
        # ========================================
        # 🔧 4-Layer Architecture Servers
        # ========================================
        # Layer 1: Edge Servers (نزدیک‌ترین - تأخیر کم)
        self.edge_servers = [
            {"position": np.array([30.0, 30.0]), "queue": 0, "capacity": 10},
            {"position": np.array([70.0, 30.0]), "queue": 0, "capacity": 10},
        ]
        
        # Layer 2: Fog Servers (متوسط - تأخیر متوسط)
        self.fog_servers = [
            {"position": np.array([50.0, 70.0]), "queue": 0, "capacity": 20},
        ]
        
        # Layer 3: Cloud Server (دورترین - تأخیر بالا، ظرفیت نامحدود)
        self.cloud_server = {
            "position": np.array([50.0, 100.0]),  # در بالای محیط
            "queue": 0,
            "capacity": 1000  # ظرفیت بسیار بالا
        }
        
        # تجمیع تمام سرورها برای دسترسی آسان
        self.all_servers = {
            1: self.edge_servers,
            2: self.fog_servers,
            3: [self.cloud_server]
        }
        
        self.dag = None
        # Task tracking lists
        self.ready_tasks = []
        self.running_tasks = []
        self.completed_tasks = []

    def reset(self, dag):
        """
        Reset environment with new DAG.
        
        Args:
            dag: dictionary با کلیدهای 'node_features', 'edge_index', etc.
        """
        self.dag = dag
        self.current_step = 0
        self.uav_position = np.array([50.0, 50.0])
        self.uav_energy = 1.0
        
        # Reset all servers
        for server_list in self.all_servers.values():
            for srv in server_list:
                srv["queue"] = 0
        
        # Initialize task lists
        num_nodes = dag['num_nodes']
        self.ready_tasks = list(range(min(3, num_nodes)))
        self.running_tasks = []
        self.completed_tasks = []
            
        return self.get_state_vector()

    def step(self, action_dict):
        """
        Execute one step in the environment (4-Layer Architecture).
        
        Args:
            action_dict: {
                "offload": int in [0, 1, 2, 3],  # 4 layers
                "cpu": float in [0, 1],
                "bandwidth": array of shape (4,),  # 4 layers
                "move": array of shape (2,)
            }
        
        Returns:
            state, reward, done, info
        """
        self.current_step += 1
        reward = 0.0
        
        # ========================================
        # 1️⃣ Offloading Decision (4 Layers)
        # ========================================
        offload_choice = action_dict.get("offload", 0)
        
        if offload_choice == 0:
            # Layer 0: Local Processing
            reward -= 5.0  # هزینه پردازش محلی (انرژی بالا)
            
        elif offload_choice == 1:
            # Layer 1: Edge Server (کمترین تأخیر)
            if len(self.edge_servers) > 0:
                # انتخاب Edge Server با کمترین صف
                best_edge = min(self.edge_servers, key=lambda s: s["queue"])
                best_edge["queue"] += 1
                reward -= 2.0  # هزینه کم
            else:
                reward -= 10.0  # خطا: Edge در دسترس نیست
                
        elif offload_choice == 2:
            # Layer 2: Fog Server (تأخیر متوسط)
            if len(self.fog_servers) > 0:
                best_fog = min(self.fog_servers, key=lambda s: s["queue"])
                best_fog["queue"] += 1
                reward -= 3.0  # هزینه متوسط
            else:
                reward -= 10.0
                
        elif offload_choice == 3:
            # Layer 3: Cloud Server (تأخیر بالا، ظرفیت نامحدود)
            self.cloud_server["queue"] += 1
            reward -= 4.0  # هزینه بالاتر به دلیل تأخیر
        
        else:
            # Invalid offload choice
            reward -= 15.0
        
        # ========================================
        # 2️⃣ CPU Allocation
        # ========================================
        cpu = action_dict.get("cpu", 0.5)
        # جریمه برای انحراف از مقدار بهینه (0.7)
        reward -= abs(cpu - 0.7) * 2.0
        
        # ========================================
        # 3️⃣ Bandwidth Allocation (4 Layers)
        # ========================================
        bw = action_dict.get("bandwidth", np.array([0.25, 0.25, 0.25, 0.25]))
        
        # بررسی صحت bandwidth (باید جمع آن 1 باشد)
        if not np.isclose(sum(bw), 1.0, atol=0.01):
            reward -= 5.0  # جریمه برای bandwidth نامعتبر
        
        # پاداش برای توزیع متوازن
        entropy = -np.sum(bw * np.log(bw + 1e-8))
        reward += entropy * 0.5  # تشویق تنوع
        
        # ========================================
        # 4️⃣ UAV Movement
        # ========================================
        move = action_dict.get("move", np.array([0.0, 0.0]))
        self.uav_position += move
        self.uav_position = np.clip(self.uav_position, 0, 100)
        
        # هزینه حرکت (انرژی)
        move_cost = np.linalg.norm(move) * 0.5
        reward -= move_cost
        
        # ========================================
        # 5️⃣ Energy Depletion
        # ========================================
        self.uav_energy -= 0.01
        if self.uav_energy < 0.2:
            reward -= 20.0  # جریمه سنگین برای انرژی کم
        
        # ========================================
        # 6️⃣ Task Completion Simulation
        # ========================================
        if len(self.ready_tasks) > 0:
            # شبیه‌سازی ساده: احتمال 30% برای اتمام task
            completed = np.random.rand() < 0.3
            if completed:
                reward += 50.0  # پاداش بزرگ برای اتمام task
                task = self.ready_tasks.pop(0)
                self.completed_tasks.append(task)
        
        # کاهش صف سرورها (شبیه‌سازی پردازش)
        for server_list in self.all_servers.values():
            for srv in server_list:
                if srv["queue"] > 0:
                    srv["queue"] = max(0, srv["queue"] - 1)
        
        # ========================================
        # 7️⃣ Check Done
        # ========================================
        done = (
            len(self.ready_tasks) == 0 and 
            len(self.running_tasks) == 0
        ) or self.current_step >= self.max_steps
        
        state = self.get_state_vector()
        
        # ساخت info dictionary با 4-Layer Metrics
        info = self._compute_metrics()
        
        return state, reward, done, info

    def get_state_vector(self):
        """
        Generate state vector from current environment state.
        
        Returns:
            numpy array of shape (537,):
                - Graph embedding: 256
                - Node embedding (pooled): 256
                - Flat state: 25
        """
        # استخراج node_features و edge_index از dictionary
        node_features = self.dag['node_features']
        edge_index = self.dag['edge_index']
        edge_attr = self.dag.get('edge_attr', None)
        
        # ساخت task_graph
        task_graph = Data(
            x=node_features.to(self.device),
            edge_index=edge_index.to(self.device),
            edge_attr=edge_attr.to(self.device) if edge_attr is not None else None
        )
        
        # دریافت embeddings
        g_emb, n_emb = self.task_encoder.get_graph_embedding(task_graph)
        
        g_emb_flat = g_emb.squeeze(0).detach().cpu().numpy().flatten()
        n_emb_pooled = n_emb.mean(dim=0).detach().cpu().numpy().flatten()
        
        # ========================================
        # Flat state (25 dims) - 4-Layer Version
        # ========================================
        flat_state = np.concatenate([
            self.uav_position,                                      # 2
            [self.uav_energy],                                      # 1
            # Queue lengths (2 edge + 1 fog + 1 cloud = 4)
            [srv["queue"] for srv in self.edge_servers],           # 2
            [srv["queue"] for srv in self.fog_servers],            # 1
            [self.cloud_server["queue"]],                          # 1
            # Server positions (sample: 2 edge + 1 fog = 6)
            self.edge_servers[0]["position"],                      # 2
            self.edge_servers[1]["position"],                      # 2
            self.fog_servers[0]["position"],                       # 2
            # Task counts
            [len(self.ready_tasks)],                               # 1
            [len(self.running_tasks)],                             # 1
            [len(self.completed_tasks)],                           # 1
            # Placeholder for future features
            np.random.rand(10)                                     # 10
        ])
        
        # Total: 256 + 256 + 25 = 537
        state = np.concatenate([g_emb_flat, n_emb_pooled, flat_state])
        
        return state
    
    def _compute_metrics(self):
        """
        محاسبه 4 معیار کیفیت برای evaluation (4-Layer Architecture).
        
        Returns:
            dict با کلیدهای:
                - delay: میانگین تاخیر وزن‌دار بر اساس لایه
                - energy_consumption: انرژی مصرف شده
                - distance: فاصله UAV از مرکز محیط
                - qos_satisfaction: نسبت task های تکمیل شده
                - layer_distribution: توزیع بار بین 4 لایه
        """
        # ========================================
        # 1️⃣ Delay (تاخیر وزن‌دار 4 لایه)
        # ========================================
        # وزن تأخیر هر لایه (ضرایب تجربی)
        delay_weights = {
            'edge': 0.1,   # Edge: کم‌ترین تأخیر
            'fog': 0.3,    # Fog: تأخیر متوسط
            'cloud': 0.6   # Cloud: بیشترین تأخیر
        }
        
        edge_delay = sum(srv["queue"] for srv in self.edge_servers) * delay_weights['edge']
        fog_delay = sum(srv["queue"] for srv in self.fog_servers) * delay_weights['fog']
        cloud_delay = self.cloud_server["queue"] * delay_weights['cloud']
        
        total_delay = edge_delay + fog_delay + cloud_delay
        
        # ========================================
        # 2️⃣ Energy Consumption
        # ========================================
        energy_consumed = 1.0 - self.uav_energy
        
        # ========================================
        # 3️⃣ Distance from Center
        # ========================================
        center = np.array([50.0, 50.0])
        distance_from_center = float(np.linalg.norm(self.uav_position - center))
        
        # ========================================
        # 4️⃣ QoS Satisfaction
        # ========================================
        total_tasks = len(self.ready_tasks) + len(self.running_tasks) + len(self.completed_tasks)
        if total_tasks > 0:
            qos_ratio = len(self.completed_tasks) / total_tasks
        else:
            qos_ratio = 0.0
        
        # ========================================
        # 5️⃣ Layer Distribution (توزیع بار)
        # ========================================
        layer_loads = {
            'edge': sum(srv["queue"] for srv in self.edge_servers),
            'fog': sum(srv["queue"] for srv in self.fog_servers),
            'cloud': self.cloud_server["queue"]
        }
        
        return {
            'delay': total_delay,
            'energy_consumption': energy_consumed,
            'distance': distance_from_center,
            'qos_satisfaction': qos_ratio,
            'layer_distribution': layer_loads,
            # اضافه: breakdown تاخیر هر لایه
            'edge_delay': edge_delay,
            'fog_delay': fog_delay,
            'cloud_delay': cloud_delay
        }


# ========================================
# 🧪 Test Function
# ========================================

def test_environment_4layer():
    """تست محیط با معماری 4 لایه"""
    print("=" * 70)
    print("🧪 Testing UAVMECEnvironment (4-Layer Architecture)")
    print("=" * 70)
    
    # ساخت DAG dummy
    dummy_dag = {
        'num_nodes': 10,
        'node_features': torch.rand(10, 64),
        'edge_index': torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        'edge_attr': None
    }
    
    env = UAVMECEnvironment(device="cpu", max_steps=20)
    state = env.reset(dummy_dag)
    
    print(f"\n✅ Environment initialized")
    print(f"   State shape: {state.shape}")
    print(f"   Edge servers: {len(env.edge_servers)}")
    print(f"   Fog servers: {len(env.fog_servers)}")
    print(f"   Cloud server: 1")
    
    # تست 4 نوع offload
    print(f"\n{'─' * 70}")
    print("📊 Testing 4-Layer Offloading:")
    print(f"{'─' * 70}")
    
    layer_names = {0: "Local", 1: "Edge", 2: "Fog", 3: "Cloud"}
    
    for layer_id in range(4):
        action = {
            "offload": layer_id,
            "cpu": 0.7,
            "bandwidth": np.array([0.25, 0.25, 0.25, 0.25]),
            "move": np.array([0.0, 0.0])
        }
        
        state, reward, done, info = env.step(action)
        
        print(f"\n🎯 Layer {layer_id} ({layer_names[layer_id]}):")
        print(f"   Reward: {reward:.2f}")
        print(f"   Delay: {info['delay']:.3f}")
        print(f"   Energy: {info['energy_consumption']:.3f}")
        print(f"   QoS: {info['qos_satisfaction']:.2%}")
        print(f"   Layer Loads: {info['layer_distribution']}")
    
    print("\n" + "=" * 70)
    print("✅ All 4 layers tested successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_environment_4layer()
