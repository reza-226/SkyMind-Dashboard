# algorithms/baselines/simple_policies.py

"""
Simple Baseline Policies for UAV Offloading (4-Layer Architecture)
===================================================================
استراتژی‌های ساده برای مقایسه با MADDPG در معماری 4 لایه:
- Layer 0: Local (UAV)
- Layer 1: Edge
- Layer 2: Fog
- Layer 3: Cloud
"""

import numpy as np
from typing import Dict, List, Union


class GreedyLocalPolicy:
    """
    استراتژی Greedy-Local با اولویت‌بندی 4 لایه
    
    منطق تصمیم‌گیری:
    1. CPU > 0.7 → Local (لوکال)
    2. CPU > 0.4 → Edge (لبه)
    3. CPU > 0.2 → Fog (مه)
    4. CPU ≤ 0.2 → Cloud (ابر)
    """
    
    def __init__(self, 
                 local_threshold=0.7, 
                 edge_threshold=0.4,
                 fog_threshold=0.2):
        self.local_threshold = local_threshold
        self.edge_threshold = edge_threshold
        self.fog_threshold = fog_threshold
        self.name = "Greedy-Local-4Layer"
    
    def select_action(self, state: np.ndarray, evaluation=False) -> Dict:
        """
        Args:
            state: numpy array با shape (537,) یا (batch, 537)
            evaluation: حالت ارزیابی (deterministic)
            
        Returns:
            action dict با کلیدهای offload, cpu, bandwidth, move
        """
        # تبدیل به batch اگر نیست
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        batch_size = state.shape[0]
        actions = []
        
        for i in range(batch_size):
            # استخراج CPU capacity (فرض: index 5)
            cpu_capacity = state[i, 5] if len(state[i]) > 5 else 0.5
            
            # تصمیم‌گیری offload بر اساس threshold
            if cpu_capacity > self.local_threshold:
                offload = 0  # Local
            elif cpu_capacity > self.edge_threshold:
                offload = 1  # Edge
            elif cpu_capacity > self.fog_threshold:
                offload = 2  # Fog
            else:
                offload = 3  # Cloud
            
            action = {
                "offload": offload,
                "cpu": 0.8,  # استفاده 80% CPU
                "bandwidth": np.array([0.25, 0.25, 0.25, 0.25]),  # توزیع یکنواخت 4 لایه
                "move": np.array([0.0, 0.0])  # بدون حرکت
            }
            actions.append(action)
        
        return actions[0] if batch_size == 1 else actions


class AlwaysLocalPolicy:
    """همیشه پردازش محلی (Local)"""
    
    def __init__(self):
        self.name = "Always-Local"
    
    def select_action(self, state: np.ndarray, evaluation=False) -> Dict:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        return {
            "offload": 0,  # Local
            "cpu": 0.9,  # استفاده بالای CPU
            "bandwidth": np.array([0.25, 0.25, 0.25, 0.25]),
            "move": np.array([0.0, 0.0])
        }


class AlwaysEdgePolicy:
    """همیشه offload به Edge"""
    
    def __init__(self):
        self.name = "Always-Edge"
    
    def select_action(self, state: np.ndarray, evaluation=False) -> Dict:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        return {
            "offload": 1,  # Edge
            "cpu": 0.6,
            "bandwidth": np.array([0.25, 0.25, 0.25, 0.25]),
            "move": np.array([0.0, 0.0])
        }


class AlwaysFogPolicy:
    """همیشه offload به Fog"""
    
    def __init__(self):
        self.name = "Always-Fog"
    
    def select_action(self, state: np.ndarray, evaluation=False) -> Dict:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        return {
            "offload": 2,  # Fog
            "cpu": 0.4,
            "bandwidth": np.array([0.25, 0.25, 0.25, 0.25]),
            "move": np.array([0.0, 0.0])
        }


class AlwaysCloudPolicy:
    """همیشه offload به Cloud"""
    
    def __init__(self):
        self.name = "Always-Cloud"
    
    def select_action(self, state: np.ndarray, evaluation=False) -> Dict:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        return {
            "offload": 3,  # Cloud (تغییر از 2 به 3)
            "cpu": 0.3,
            "bandwidth": np.array([0.25, 0.25, 0.25, 0.25]),  # 4 لایه
            "move": np.array([0.0, 0.0])
        }


class RandomPolicy:
    """سیاست تصادفی برای baseline"""
    
    def __init__(self, action_dim=9):  # 1 + 1 + 4 + 2 + 1 (offload)
        self.action_dim = action_dim
        self.name = "Random-4Layer"
    
    def select_action(self, state: np.ndarray, evaluation=False) -> Dict:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # انتخاب تصادفی از 4 لایه
        offload = np.random.randint(0, 4)  # 0, 1, 2, 3
        
        # bandwidth تصادفی که جمع آن‌ها 1 شود
        bw = np.random.dirichlet(np.ones(4))
        
        return {
            "offload": offload,
            "cpu": np.random.uniform(0.3, 0.9),
            "bandwidth": bw,
            "move": np.random.uniform(-1, 1, size=2)
        }


class LoadBalancingPolicy:
    """
    سیاست توزیع بار (Load Balancing) بین 4 لایه
    
    منطق: تصمیم بر اساس بار فعلی سیستم
    """
    
    def __init__(self):
        self.name = "Load-Balancing-4Layer"
        self.layer_loads = np.zeros(4)  # بار هر لایه
        self.step_count = 0
    
    def select_action(self, state: np.ndarray, evaluation=False) -> Dict:
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # انتخاب لایه‌ای با کمترین بار
        offload = int(np.argmin(self.layer_loads))
        
        # به‌روزرسانی بار
        self.layer_loads[offload] += 1
        self.step_count += 1
        
        # ریست هر 100 گام
        if self.step_count % 100 == 0:
            self.layer_loads = np.zeros(4)
        
        return {
            "offload": offload,
            "cpu": 0.6,
            "bandwidth": np.array([0.25, 0.25, 0.25, 0.25]),
            "move": np.array([0.0, 0.0])
        }


# ========================================
# Test Function
# ========================================

def test_policies():
    """تست تمام policies برای معماری 4 لایه"""
    print("=" * 70)
    print("🧪 Testing Simple Policies (4-Layer Architecture)")
    print("=" * 70)
    
    # State dummy
    dummy_state = np.random.rand(537)
    
    policies = [
        GreedyLocalPolicy(),
        AlwaysLocalPolicy(),
        AlwaysEdgePolicy(),
        AlwaysFogPolicy(),
        AlwaysCloudPolicy(),
        RandomPolicy(),
        LoadBalancingPolicy()
    ]
    
    print(f"\n📊 Input state shape: {dummy_state.shape}")
    print(f"📊 Number of layers: 4 (Local, Edge, Fog, Cloud)\n")
    
    for policy in policies:
        print(f"{'─' * 70}")
        print(f"🔧 Policy: {policy.name}")
        print(f"{'─' * 70}")
        
        # تست با evaluation=False (Training)
        action = policy.select_action(dummy_state, evaluation=False)
        print(f"   🎯 Training Mode:")
        print(f"      - Offload layer: {action['offload']} ", end="")
        layer_names = {0: "Local", 1: "Edge", 2: "Fog", 3: "Cloud"}
        print(f"({layer_names[action['offload']]})")
        print(f"      - CPU usage: {action['cpu']:.2f}")
        print(f"      - Bandwidth: {action['bandwidth']}")
        print(f"      - Movement: {action['move']}")
        
        # تست با evaluation=True (Evaluation)
        action = policy.select_action(dummy_state, evaluation=True)
        print(f"   🎯 Evaluation Mode:")
        print(f"      - Offload layer: {action['offload']} ", end="")
        print(f"({layer_names[action['offload']]})")
        print(f"      ✅ Success!")
        print()
    
    print("=" * 70)
    print("✅ All policies tested successfully!")
    print("=" * 70)


def test_policy_distribution():
    """تست توزیع تصمیمات Random Policy"""
    print("\n" + "=" * 70)
    print("📊 Testing Random Policy Distribution (1000 samples)")
    print("=" * 70)
    
    policy = RandomPolicy()
    dummy_state = np.random.rand(537)
    
    offload_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for _ in range(1000):
        action = policy.select_action(dummy_state)
        offload_counts[action['offload']] += 1
    
    print("\n🎲 Offload Distribution:")
    layer_names = {0: "Local", 1: "Edge", 2: "Fog", 3: "Cloud"}
    for layer, count in offload_counts.items():
        percentage = (count / 1000) * 100
        print(f"   {layer_names[layer]:6s}: {count:4d} ({percentage:5.1f}%)")
    
    print("=" * 70)


if __name__ == "__main__":
    test_policies()
    test_policy_distribution()
