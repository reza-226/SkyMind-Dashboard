"""
تست مستقل GNN (بدون وابستگی به بقیه پروژه)
"""
import torch
from torch_geometric.data import Data
import sys
import os

# اضافه کردن مسیر پروژه
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn.task_encoder import GNNTaskEncoder

def test_single_graph():
    """تست 1: یک گراف ساده"""
    print("\n" + "=" * 60)
    print("TEST 1: Single Graph")
    print("=" * 60)
    
    model = GNNTaskEncoder(input_dim=4, embedding_dim=64, num_gat_layers=2)
    
    # گراف با 5 nodes
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    graph_emb, node_emb = model(data)
    
    print(f"✅ Graph embedding: {graph_emb.shape}")
    print(f"✅ Node embeddings: {node_emb.shape}")
    
    assert graph_emb.shape == (1, 64), "Graph embedding shape is wrong!"
    assert node_emb.shape == (5, 64), "Node embeddings shape is wrong!"
    print("✅ Test 1 PASSED\n")

def test_different_sizes():
    """تست 2: گراف‌های با اندازه‌های مختلف"""
    print("=" * 60)
    print("TEST 2: Different Graph Sizes")
    print("=" * 60)
    
    model = GNNTaskEncoder(input_dim=4, embedding_dim=32, num_gat_layers=3)
    
    for num_nodes in [3, 7, 10, 15]:
        x = torch.randn(num_nodes, 4)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        data = Data(x=x, edge_index=edge_index)
        
        graph_emb, node_emb = model(data)
        
        assert graph_emb.shape == (1, 32), f"Failed for {num_nodes} nodes"
        assert node_emb.shape == (num_nodes, 32), f"Failed for {num_nodes} nodes"
        
        print(f"✅ {num_nodes} nodes: graph_emb={graph_emb.shape}, node_emb={node_emb.shape}")
    
    print("✅ Test 2 PASSED\n")

def test_integration_with_drl():
    """تست 3: شبیه‌سازی ادغام با DRL Agent"""
    print("=" * 60)
    print("TEST 3: Integration Simulation")
    print("=" * 60)
    
    embedding_dim = 64
    env_state_dim = 12
    
    # GNN
    gnn = GNNTaskEncoder(input_dim=4, embedding_dim=embedding_dim, num_gat_layers=2)
    
    # ساختن گراف
    x = torch.randn(7, 4)
    edge_index = torch.randint(0, 7, (2, 14))
    data = Data(x=x, edge_index=edge_index)
    
    # Forward pass
    graph_embedding, _ = gnn(data)  # (1, 64)
    
    # شبیه‌سازی env_state
    env_state = torch.randn(env_state_dim)  # (12,)
    env_state = env_state.unsqueeze(0)  # (1, 12) برای match با batch
    
    # ادغام
    combined = torch.cat([graph_embedding, env_state], dim=1)  # (1, 76)
    
    print(f"✅ Graph embedding: {graph_embedding.shape}")
    print(f"✅ Environment state: {env_state.shape}")
    print(f"✅ Combined: {combined.shape}")
    
    assert combined.shape == (1, embedding_dim + env_state_dim)
    print("✅ Test 3 PASSED\n")

if __name__ == "__main__":
    print("\n" + "🧪" * 30)
    print("GNN STANDALONE TESTS")
    print("🧪" * 30 + "\n")
    
    try:
        test_single_graph()
        test_different_sizes()
        test_integration_with_drl()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
