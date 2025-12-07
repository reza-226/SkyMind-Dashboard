"""
Test script for GNN Task Encoder
اسکریپت تست برای رمزگذار GNN
"""

import torch
from models.gnn.task_encoder import GNNTaskEncoder
from utils.graph_utils import TaskDAG, TaskNode, convert_dag_to_pyg_data


def test_gnn_encoder():
    """تست عملکرد GNN Task Encoder"""
    
    print("=" * 50)
    print("🧪 Testing GNN Task Encoder")
    print("=" * 50)
    
    # ایجاد یک DAG تست
    print("\n📊 Generating test DAG...")
    dag = TaskDAG(dag_id="test_dag_001")
    
    # اضافه کردن 10 task با پارامترهای صحیح
    for i in range(10):
        task = TaskNode(
            task_id=i,
            data_size=torch.rand(1).item() * 100,          # 0-100 MB
            comp_requirement=torch.rand(1).item() * 1000,  # 0-1000 CPU cycles
            deadline=torch.rand(1).item() * 10,            # 0-10 seconds
            priority=torch.rand(1).item()                  # 0-1
        )
        dag.add_task(task)
    
    # اضافه کردن وابستگی‌ها
    dependencies = [
        (0, 1), (0, 2), (1, 3), (2, 3), (2, 4),
        (3, 5), (4, 5), (4, 6), (5, 7), (6, 7),
        (7, 8), (7, 9), (8, 9), (1, 6)
    ]
    for src, dst in dependencies:
        dag.add_dependency(src, dst)
    
    ready_tasks = dag.get_ready_tasks()
    print(f"✅ Generated DAG with {len(dag.nodes)} tasks and {len(dependencies)} dependencies")
    print(f"   Ready tasks: {ready_tasks}")
    
    # تبدیل به PyTorch Geometric Data
    print("\n🔄 Converting to PyTorch Geometric...")
    task_graph = convert_dag_to_pyg_data(dag)
    
    print(f"✅ Graph structure:")
    print(f"   - Nodes: {task_graph.num_nodes}")
    print(f"   - Edges: {task_graph.num_edges}")
    print(f"   - Node features: {task_graph.x.shape}")
    print(f"   - Edge features: {task_graph.edge_attr.shape}")
    
    # ساخت encoder
    print("\n🏗️ Building GNN Encoder...")
    encoder = GNNTaskEncoder(
        node_feature_dim=9,
        edge_feature_dim=3,
        hidden_dim=256,
        embedding_dim=256,
        num_gat_layers=3,
        num_heads=4,
        dropout=0.1
    )
    
    num_params = encoder.count_parameters()
    print(f"✅ Encoder created:")
    print(f"   - Parameters: {num_params:,}")
    
    # Forward pass
    print("\n🚀 Running forward pass...")
    with torch.no_grad():
        task_embeddings, critical_scores = encoder(task_graph)
    
    print(f"✅ Forward pass successful!")
    print(f"   - Task embeddings: {task_embeddings.shape}")
    print(f"   - Critical scores: {critical_scores.shape}")
    
    # تست graph embedding
    print("\n🌐 Testing graph embedding...")
    with torch.no_grad():
        graph_embedding = encoder.get_graph_embedding(task_graph)
    print(f"✅ Graph embedding: {graph_embedding.shape}")
    
    # تست critical path detection
    print("\n🎯 Testing critical path detection...")
    with torch.no_grad():
        critical_mask = encoder.get_critical_path(task_graph, threshold=0.5)
    critical_tasks = torch.where(critical_mask)[0].tolist()
    print(f"✅ Critical tasks detected: {critical_tasks}")
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    test_gnn_encoder()
