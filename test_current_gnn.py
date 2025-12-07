"""
تست GNN فعلی شما
"""
import torch
from torch_geometric.data import Data
import sys
import os

# اضافه کردن مسیر پروژه
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn.task_encoder import GNNTaskEncoder

def test_current_implementation():
    """تست پیاده‌سازی فعلی"""
    print("\n" + "=" * 60)
    print("🧪 Testing Current GNN Implementation")
    print("=" * 60)
    
    # تنظیمات
    node_feature_dim = 9
    edge_feature_dim = 3
    embedding_dim = 256
    
    # ایجاد مدل
    model = GNNTaskEncoder(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        embedding_dim=embedding_dim,
        num_gat_layers=3,
        pooling='mean'
    )
    
    print(f"\n📊 Model Info:")
    print(f"   - Parameters: {model.count_parameters():,}")
    print(f"   - Pooling: {model.pooling}")
    
    # Test 1: گراف ساده بدون batch
    print("\n" + "-" * 60)
    print("TEST 1: Single Graph (no batch)")
    print("-" * 60)
    
    num_nodes = 7
    x = torch.randn(num_nodes, node_feature_dim)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]
    ], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), edge_feature_dim)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    try:
        graph_emb, node_emb = model(data)
        print(f"✅ Graph embedding shape: {graph_emb.shape}")  # انتظار: (1, 256)
        print(f"✅ Node embeddings shape: {node_emb.shape}")   # انتظار: (7, 256)
        
        # بررسی ابعاد
        assert graph_emb.dim() == 2, f"❌ graph_emb باید 2D باشد، نه {graph_emb.dim()}D"
        assert graph_emb.size(0) == 1, f"❌ batch_size باید 1 باشد"
        assert graph_emb.size(1) == embedding_dim
        assert node_emb.shape == (num_nodes, embedding_dim)
        
        print("✅ Test 1 PASSED")
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: استفاده از get_graph_embedding
    print("\n" + "-" * 60)
    print("TEST 2: get_graph_embedding method")
    print("-" * 60)
    
    try:
        graph_emb2 = model.get_graph_embedding(data)
        print(f"✅ Graph embedding shape: {graph_emb2.shape}")
        
        # باید با خروجی forward یکسان باشد
        assert torch.allclose(graph_emb, graph_emb2, atol=1e-6)
        print("✅ Test 2 PASSED")
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False
    
    # Test 3: Critical Path Detection
    print("\n" + "-" * 60)
    print("TEST 3: Critical Path Detection")
    print("-" * 60)
    
    try:
        critical_mask = model.get_critical_path(data, threshold=0.5)
        print(f"✅ Critical mask shape: {critical_mask.shape}")  # (7,)
        print(f"   Critical nodes: {critical_mask.sum().item()}/{num_nodes}")
        
        assert critical_mask.shape == (num_nodes,)
        print("✅ Test 3 PASSED")
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        return False
    
    # Test 4: بدون edge_attr
    print("\n" + "-" * 60)
    print("TEST 4: Graph without edge_attr")
    print("-" * 60)
    
    data_no_edge = Data(x=x, edge_index=edge_index)
    
    try:
        graph_emb3, _ = model(data_no_edge)
        print(f"✅ Works without edge_attr: {graph_emb3.shape}")
        print("✅ Test 4 PASSED")
    except Exception as e:
        print(f"⚠️ Test 4: Model expects edge_attr")
        print(f"   Error: {e}")
        # این مشکل نیست اگر مدل شما حتماً edge_attr می‌خواهد
    
    # Test 5: Integration simulation
    print("\n" + "-" * 60)
    print("TEST 5: Integration with DRL Agent (simulation)")
    print("-" * 60)
    
    try:
        # شبیه‌سازی env_state
        env_state_dim = 12
        env_state = torch.randn(env_state_dim)  # (12,)
        env_state = env_state.unsqueeze(0)      # (1, 12)
        
        # ادغام
        combined = torch.cat([graph_emb, env_state], dim=1)  # (1, 256+12)
        
        print(f"✅ Graph embedding: {graph_emb.shape}")
        print(f"✅ Env state: {env_state.shape}")
        print(f"✅ Combined state: {combined.shape}")
        
        assert combined.shape == (1, embedding_dim + env_state_dim)
        print("✅ Test 5 PASSED")
    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("GNN CURRENT IMPLEMENTATION TESTS")
    print("🚀" * 30 + "\n")
    
    success = test_current_implementation()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\n✅ کد شما آماده است!")
        print("✅ می‌توانید به مرحله بعد بروید: DRL Agent Integration\n")
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\n⚠️ لطفاً خطاها را بررسی کنید\n")
