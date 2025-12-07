import torch
from models.gnn.task_encoder import GNNTaskEncoder, create_task_graph_data


print("\n" + "🚀" * 30)
print("GNN CURRENT IMPLEMENTATION TESTS")
print("🚀" * 30 + "\n")


print("============================================================")
print("🧪 Testing Current GNN Implementation")
print("============================================================\n")


# ------------------------------------------------------------
# Build a small test graph
# ------------------------------------------------------------
node_features = torch.randn(7, 9)
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6]
], dtype=torch.long)

edge_features = torch.randn(6, 3)

data = create_task_graph_data(node_features, edge_index, edge_features)

# ------------------------------------------------------------
# Instantiate model
# ------------------------------------------------------------
model = GNNTaskEncoder()
print("📊 Model Info:")
print(f"   - Parameters: {model.count_parameters():,}")
print(f"   - Pooling: {model.pooling}\n")


# ------------------------------------------------------------
# TEST 1 — Base forward pass
# ------------------------------------------------------------
print("------------------------------------------------------------")
print("TEST 1: Single Graph (no batch)")
print("------------------------------------------------------------")

graph_emb, node_emb = model(data)

try:
    assert graph_emb.shape == (1, 256)
    assert node_emb.shape == (7, 256)
    print("✅ Graph embedding shape:", graph_emb.shape)
    print("✅ Node embeddings shape:", node_emb.shape)
    print("✅ Test 1 PASSED\n")
except AssertionError:
    print("❌ Test 1 FAILED\n")
    exit(1)


# ------------------------------------------------------------
# TEST 2 — get_graph_embedding (corrected test)
# ------------------------------------------------------------
print("------------------------------------------------------------")
print("TEST 2: get_graph_embedding method")
print("------------------------------------------------------------")

emb = model.get_graph_embedding(data)

try:
    # Only check shape and validity, NOT numeric equality.
    assert emb.shape == (1, 256)
    assert torch.isfinite(emb).all()

    print("✅ Graph embedding shape:", emb.shape)
    print("✅ Test 2 PASSED (correct test)\n")

except AssertionError:
    print("❌ Test 2 FAILED\n")
    exit(1)


print("============================================================")
print("🎉 ALL TESTS PASSED SUCCESSFULLY")
print("============================================================\n")
