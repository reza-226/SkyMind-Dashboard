#  models\dag.py
import torch
import numpy as np
import networkx as nx

class DAG:
    """
    Directed Acyclic Graph for modeling task dependencies in UAV offloading.
    """
    
    @staticmethod
    def generate_random_dag(num_nodes=10, edge_prob=0.3, device='cpu'):
        """
        Generate a random DAG with given number of nodes.
        
        Args:
            num_nodes: Number of tasks/nodes in the DAG
            edge_prob: Probability of creating edges between nodes
            device: torch device ('cpu' or 'cuda')
            
        Returns:
            dict containing:
                - adjacency_matrix: torch.Tensor (num_nodes, num_nodes)
                - node_features: torch.Tensor (num_nodes, feature_dim)
                - edge_index: torch.Tensor (2, num_edges) for GNN
        """
        # Create random DAG using NetworkX
        graph = nx.DiGraph()
        graph.add_nodes_from(range(num_nodes))
        
        # Add edges ensuring no cycles (only from lower to higher index)
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.random() < edge_prob:
                    graph.add_edge(i, j)
                    edges.append([i, j])
        
        # Generate node features (computation load, data size, etc.)
        node_features = torch.randn(num_nodes, 8, device=device)  # 8 features per node
        node_features[:, 0] = torch.rand(num_nodes, device=device) * 2.0 + 0.5  # computation: 0.5-2.5 GHz
        node_features[:, 1] = torch.rand(num_nodes, device=device) * 1.0 + 0.1  # data size: 0.1-1.1 MB
        
        # Convert to adjacency matrix
        adj_matrix = torch.tensor(
            nx.adjacency_matrix(graph).todense(), 
            dtype=torch.float32,
            device=device
        )
        
        # Convert to edge_index format for GNN
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        return {
            'adjacency_matrix': adj_matrix,
            'node_features': node_features,
            'edge_index': edge_index,
            'num_nodes': num_nodes,
            'num_edges': len(edges)
        }
    
    @staticmethod
    def get_topological_order(adjacency_matrix):
        """
        Get topological ordering from adjacency matrix.
        
        Args:
            adjacency_matrix: torch.Tensor (N, N)
            
        Returns:
            list: Topological order of nodes
        """
        adj_np = adjacency_matrix.cpu().numpy()
        graph = nx.from_numpy_array(adj_np, create_using=nx.DiGraph)
        
        try:
            return list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # If has cycles, return sequential order
            return list(range(len(adjacency_matrix)))
    
    @staticmethod
    def validate_dag(adjacency_matrix):
        """
        Check if the adjacency matrix represents a valid DAG (no cycles).
        
        Args:
            adjacency_matrix: torch.Tensor (N, N)
            
        Returns:
            bool: True if valid DAG
        """
        adj_np = adjacency_matrix.cpu().numpy()
        graph = nx.from_numpy_array(adj_np, create_using=nx.DiGraph)
        return nx.is_directed_acyclic_graph(graph)


# Alias for backward compatibility
def generate_random_dag(num_nodes=10, device='cpu'):
    """Legacy function wrapper."""
    return DAG.generate_random_dag(num_nodes=num_nodes, device=device)
