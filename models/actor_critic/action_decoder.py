import torch
import torch.nn.functional as F
import numpy as np

class ActionDecoder:
    """
    Convert actor outputs into environment actions.
    Returns numpy array format expected by UAVEnvironment.
    
    Action format (11-dim):
        [offload_onehot(5), cpu(1), bandwidth(3), movement(2)]
    """

    def __init__(self, max_movement_step=5.0):
        self.max_step = max_movement_step

    def decode_batch(self, offload_logits_batch):
        """
        Convert batch of offload logits to environment action format.
        
        Args:
            offload_logits_batch: numpy array (n_agents, 5) - offload logits
            
        Returns:
            actions_dict: {agent_id: numpy_array(11,)} matching UAVEnvironment.step()
        """
        n_agents = offload_logits_batch.shape[0]
        actions_dict = {}
        
        for i in range(n_agents):
            # ✅ محیط یک آرایه 11 بعدی می‌خواد:
            # [offload_onehot(5), cpu(1), bandwidth(3), movement(2)]
            action_array = np.zeros(11, dtype=np.float32)
            
            # Offload decision: One-hot encoding (indices 0-4)
            offload_idx = int(np.argmax(offload_logits_batch[i]))
            action_array[offload_idx] = 1.0
            
            # CPU allocation (index 5): range [0, 1] → [-1, 1] for env
            action_array[5] = 0.0  # پیش‌فرض 0.5 در محیط بعد از تبدیل
            
            # Bandwidth allocation (indices 6-8): range [0, 1] → [-1, 1] for env
            action_array[6:9] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            # Movement (indices 9-10): range [-1, 1]
            action_array[9:11] = np.array([0.0, 0.0], dtype=np.float32)
            
            actions_dict[f'uav_{i}'] = action_array
        
        return actions_dict
    
    def decode(self, offload_logits, cont=None):
        """
        Single action decoding with continuous parameters.
        
        Args:
            offload_logits: (B, 5) - offload decisions
            cont: (B, 6) - [cpu, bw1, bw2, bw3, move_x, move_y] in range [-1, 1]

        Returns:
            List of action arrays (11-dim each)
        """
        B = offload_logits.size(0)
        
        # Convert to numpy
        if isinstance(offload_logits, torch.Tensor):
            offload_logits = offload_logits.detach().cpu().numpy()
        if cont is not None and isinstance(cont, torch.Tensor):
            cont = cont.detach().cpu().numpy()
        
        actions = []
        for i in range(B):
            action_array = np.zeros(11, dtype=np.float32)
            
            # One-hot offload decision
            offload_idx = int(np.argmax(offload_logits[i]))
            action_array[offload_idx] = 1.0
            
            if cont is not None:
                # Use provided continuous values
                action_array[5] = cont[i, 0]      # CPU
                action_array[6:9] = cont[i, 1:4]  # Bandwidth
                action_array[9:11] = cont[i, 4:6] # Movement
            else:
                # Default values
                action_array[5] = 0.0
                action_array[6:9] = [0.0, 0.0, 0.0]
                action_array[9:11] = [0.0, 0.0]
            
            actions.append(action_array)
        
        return actions
    
    def decode_full(self, offload_logits, continuous_values):
        """
        Full decoding with all continuous parameters.
        
        Args:
            offload_logits: (B, 5) - offload decisions
            continuous_values: (B, 6) - continuous actions in [-1, 1]
            
        Returns:
            List of 11-dim action arrays ready for env.step()
        """
        return self.decode(offload_logits, continuous_values)
