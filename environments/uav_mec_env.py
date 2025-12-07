# environments/uav_mec_env.py (نسخه اصلاح شده)

import numpy as np
import torch
from typing import Dict, Tuple, Optional
import networkx as nx


class UAVMECEnvironment:
    """
    محیط شبیه‌سازی UAV-MEC با معماری 4 لایه
    Layers: Local, Edge, Fog, Cloud
    """
    
    def __init__(
        self,
        num_uavs: int = 5,
        num_devices: int = 10,
        num_edge_servers: int = 2,
        grid_size: float = 1000.0,
        max_steps: int = 100
    ):
        self.num_uavs = num_uavs
        self.num_devices = num_devices
        self.num_edge_servers = num_edge_servers
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        
        # تنظیمات منابع
        self.local_capacity = 1.0  # GHz
        self.edge_capacity = 5.0   # GHz per server
        self.fog_capacity = 10.0   # GHz
        self.cloud_capacity = 100.0  # GHz (unlimited)
        
        # تنظیمات تأخیر
        self.local_delay = 0.0
        self.edge_delay = 0.1      # وزن تأخیر
        self.fog_delay = 0.3       # وزن تأخیر
        self.cloud_delay = 0.6     # وزن تأخیر
        
        # وضعیت محیط
        self.uav_positions = None
        self.uav_velocities = None
        self.device_positions = None
        self.device_demands = None
        self.edge_positions = None
        self.task_graph = None
        
        # منابع لایه‌ها
        self.local_resources = None
        self.edge_resources = None
        self.fog_resources = None
        self.cloud_resources = None
        
        # محاسبه بعد state بعد از مقداردهی اولیه
        self.state_dim = None
        
        # محاسبه بعد action
        # action = [offload_layer (0-3), bandwidth_allocation (4 values)]
        self.action_dim = 5
    
    def reset(self, task_graph: Optional[Dict] = None) -> np.ndarray:
        """ریست محیط"""
        self.current_step = 0
        
        # مقداردهی تصادفی UAVها
        self.uav_positions = np.random.rand(self.num_uavs, 3) * self.grid_size
        self.uav_velocities = np.random.randn(self.num_uavs, 3) * 10.0
        
        # مقداردهی دستگاه‌ها
        self.device_positions = np.random.rand(self.num_devices, 2) * self.grid_size
        self.device_demands = np.random.rand(self.num_devices) * 5.0
        
        # مقداردهی Edge servers
        self.edge_positions = np.random.rand(self.num_edge_servers, 3) * self.grid_size
        
        # منابع اولیه
        self.local_resources = np.ones((self.num_uavs, 2)) * self.local_capacity
        self.edge_resources = np.ones((self.num_edge_servers, 2)) * self.edge_capacity
        self.fog_resources = np.array([self.fog_capacity, self.fog_capacity])
        self.cloud_resources = np.array([self.cloud_capacity, self.cloud_capacity])
        
        # ذخیره task graph
        self.task_graph = task_graph
        
        # محاسبه state_dim از state واقعی
        state = self._get_state()
        self.state_dim = state.shape[0]
        
        return state
    
    def _get_state(self) -> np.ndarray:
        """ساخت state vector"""
        state_components = []
        
        # 1. UAV positions (num_uavs × 3)
        state_components.append(self.uav_positions.flatten())
        
        # 2. UAV velocities (num_uavs × 3)
        state_components.append(self.uav_velocities.flatten())
        
        # 3. Local resources (num_uavs × 2)
        state_components.append(self.local_resources.flatten())
        
        # 4. Edge resources replicated for each UAV (num_uavs × 2)
        edge_flat = self.edge_resources.flatten()  # (num_edge_servers × 2)
        edge_repeated = np.tile(edge_flat, (self.num_uavs // self.num_edge_servers) + 1)[:self.num_uavs * 2]
        state_components.append(edge_repeated)
        
        # 5. Fog resources replicated for each UAV (num_uavs × 2)
        fog_repeated = np.tile(self.fog_resources, self.num_uavs)
        state_components.append(fog_repeated)
        
        # 6. Cloud resources replicated for each UAV (num_uavs × 2)
        cloud_repeated = np.tile(self.cloud_resources, self.num_uavs)
        state_components.append(cloud_repeated)
        
        # 7. Device positions (num_devices × 2)
        state_components.append(self.device_positions.flatten())
        
        # 8. Device demands (num_devices × 1)
        state_components.append(self.device_demands.flatten())
        
        # 9. Edge server positions (num_edge_servers × 3)
        state_components.append(self.edge_positions.flatten())
        
        # 10. Edge server resources (num_edge_servers × 2)
        state_components.append(self.edge_resources.flatten())
        
        # 11. Fog resources (2)
        state_components.append(self.fog_resources)
        
        # 12. Cloud resources (2)
        state_components.append(self.cloud_resources)
        
        state = np.concatenate(state_components)
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """اجرای یک گام"""
        self.current_step += 1
        
        # پردازش action
        offload_layer = int(action[0])  # 0=Local, 1=Edge, 2=Fog, 3=Cloud
        bandwidth_allocation = action[1:5]  # توزیع bandwidth
        
        # محاسبه reward
        reward = self._calculate_reward(offload_layer, bandwidth_allocation)
        
        # به‌روزرسانی وضعیت
        self._update_environment()
        
        # بررسی پایان episode
        done = self.current_step >= self.max_steps
        
        # اطلاعات اضافی
        info = {
            'offload_layer': offload_layer,
            'bandwidth': bandwidth_allocation,
            'step': self.current_step
        }
        
        next_state = self._get_state()
        return next_state, reward, done, info
    
    def _calculate_reward(self, offload_layer: int, bandwidth: np.ndarray) -> float:
        """محاسبه reward بر اساس لایه انتخاب شده"""
        
        # بررسی اعتبار offload_layer
        if offload_layer < 0 or offload_layer > 3:
            return -100.0  # جریمه برای انتخاب نامعتبر
        
        # بررسی اعتبار bandwidth
        if np.any(bandwidth < 0) or np.any(bandwidth > 1):
            return -50.0
        
        # محاسبه تأخیر بر اساس لایه
        if offload_layer == 0:  # Local
            delay_penalty = self.local_delay
            capacity = self.local_capacity
        elif offload_layer == 1:  # Edge
            delay_penalty = self.edge_delay
            capacity = self.edge_capacity
        elif offload_layer == 2:  # Fog
            delay_penalty = self.fog_delay
            capacity = self.fog_capacity
        else:  # Cloud
            delay_penalty = self.cloud_delay
            capacity = self.cloud_capacity
        
        # محاسبه مصرف انرژی
        energy_cost = np.sum(bandwidth) * 0.1
        
        # محاسبه بهره‌وری منابع
        resource_efficiency = capacity / (capacity + 1.0)
        
        # reward نهایی
        reward = (
            100.0 * resource_efficiency -
            50.0 * delay_penalty -
            20.0 * energy_cost
        )
        
        return reward
    
    def _update_environment(self):
        """به‌روزرسانی وضعیت محیط"""
        # حرکت UAVها
        self.uav_positions += self.uav_velocities * 0.1
        
        # محدود کردن به grid
        self.uav_positions = np.clip(self.uav_positions, 0, self.grid_size)
        
        # به‌روزرسانی تصادفی منابع
        noise = np.random.randn(*self.local_resources.shape) * 0.05
        self.local_resources = np.clip(
            self.local_resources + noise,
            0.5 * self.local_capacity,
            1.5 * self.local_capacity
        )
    
    def render(self):
        """نمایش وضعیت محیط"""
        print(f"\n{'='*70}")
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"UAVs: {self.num_uavs}, Devices: {self.num_devices}")
        print(f"Edge Servers: {self.num_edge_servers}")
        print(f"{'='*70}")


def test_environment_4layer():
    """تست محیط 4-لایه با GNN"""
    print("\n" + "="*70)
    print("🧪 Testing 4-Layer UAV-MEC Environment")
    print("="*70)
    
    num_uavs = 5
    num_devices = 10
    num_edge_servers = 2
    
    env = UAVMECEnvironment(
        num_uavs=num_uavs,
        num_devices=num_devices,
        num_edge_servers=num_edge_servers,
        grid_size=1000.0,
        max_steps=100
    )
    print("✅ Environment created")
    
    # ساخت DAG ساده
    num_nodes = 10
    node_features = torch.randn(num_nodes, 8)
    
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ], dtype=torch.long)
    
    # ✅ بدون edge features (مطابق با task_encoder.py)
    dag = {
        'num_nodes': num_nodes,
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_attr': None  # ✅ حذف edge features
    }
    
    print(f"✅ DAG created: {num_nodes} nodes, {edge_index.size(1)} edges")
    print(f"   Node features: {node_features.shape}")
    print(f"   Edge features: None")
    
    # Reset
    print("\n" + "="*70)
    print("🔄 Resetting Environment...")
    print("="*70)
    
    state = env.reset(task_graph=dag)
    print(f"✅ State shape: {state.shape}")
    print(f"✅ State dimension: {env.state_dim}")
    
    # تحلیل ابعاد state
    print("\n📊 State Dimension Breakdown:")
    idx = 0
    components = [
        ("UAV positions", num_uavs * 3),
        ("UAV velocities", num_uavs * 3),
        ("Local resources", num_uavs * 2),
        ("Edge resources (replicated)", num_uavs * 2),
        ("Fog resources (replicated)", num_uavs * 2),
        ("Cloud resources (replicated)", num_uavs * 2),
        ("Device positions", num_devices * 2),
        ("Device demands", num_devices * 1),
        ("Edge server positions", num_edge_servers * 3),
        ("Edge server resources", num_edge_servers * 2),
        ("Fog resources", 2),
        ("Cloud resources", 2)
    ]
    
    total = 0
    for name, size in components:
        print(f"  {name:30s}: {size:3d} dims")
        total += size
    
    print(f"  {'-'*40}")
    print(f"  {'Total':30s}: {total:3d} dims")
    print(f"  {'Actual':30s}: {state.shape[0]:3d} dims")
    
    # تست action
    print("\n" + "="*70)
    print("🎯 Testing Step Function (4-Layer)...")
    print("="*70)
    
    # Action: [offload_layer, bandwidth_4layers]
    test_cases = [
        ("Local Processing", np.array([0, 1.0, 0.0, 0.0, 0.0])),
        ("Edge Processing", np.array([1, 0.0, 1.0, 0.0, 0.0])),
        ("Fog Processing", np.array([2, 0.0, 0.0, 1.0, 0.0])),
        ("Cloud Processing", np.array([3, 0.0, 0.0, 0.0, 1.0])),
        ("Mixed Allocation", np.array([1, 0.4, 0.3, 0.2, 0.1]))
    ]
    
    for test_name, action in test_cases:
        next_state, reward, done, info = env.step(action)
        print(f"\n  {test_name}:")
        print(f"    Offload: Layer {int(action[0])} ({'Local' if action[0]==0 else 'Edge' if action[0]==1 else 'Fog' if action[0]==2 else 'Cloud'})")
        print(f"    Bandwidth: {action[1:]}")
        print(f"    Reward: {reward:.2f}")
        print(f"    Done: {done}")
    
    # تست سناریوهای خطا
    print("\n" + "="*70)
    print("⚠️  Testing Error Scenarios...")
    print("="*70)
    
    error_cases = [
        ("Invalid Offload", np.array([5, 0.5, 0.3, 0.1, 0.1])),
        ("Invalid Bandwidth", np.array([1, -0.5, 0.5, 0.5, 0.5]))
    ]
    
    for test_name, action in error_cases:
        next_state, reward, done, info = env.step(action)
        print(f"\n  {test_name}:")
        print(f"    Action: {action}")
        print(f"    Reward (penalty): {reward:.2f}")
    
    # تست چند step متوالی
    print("\n" + "="*70)
    print("🔄 Testing Multiple Sequential Steps...")
    print("="*70)
    
    env.reset(task_graph=dag)
    for i in range(5):
        offload = np.random.randint(0, 4)
        bw = np.random.dirichlet(np.ones(4))
        action = np.concatenate([[offload], bw])
        
        next_state, reward, done, info = env.step(action)
        layer_name = ['Local', 'Edge', 'Fog', 'Cloud'][offload]
        print(f"  Step {i+1}: Layer={layer_name}, Reward={reward:.2f}, Done={done}")
        
        if done:
            print("  ✅ Episode finished!")
            break
    
    print("\n" + "="*70)
    print("✅ All 4-Layer Tests Passed Successfully!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"  • Environment: 4-Layer (Local, Edge, Fog, Cloud)")
    print(f"  • State dimension: {state.shape[0]}")
    print(f"  • Action dimension: 5 (1 offload + 4 bandwidth)")
    print(f"  • UAVs: {num_uavs}")
    print(f"  • Devices: {num_devices}")
    print(f"  • Edge Servers: {num_edge_servers}")
    print("="*70)


if __name__ == "__main__":
    test_environment_4layer()
