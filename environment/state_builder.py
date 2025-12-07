import numpy as np

class StateBuilder:
    """Build 537-dim state vector"""
    
    def __init__(self):
        self.graph_dim = 256
        self.node_dim = 256
        self.flat_dim = 25
    
    def build_state(self, graph_features, node_features, flat_features):
        state = np.concatenate([
            graph_features,
            node_features,
            flat_features
        ])
        return state
