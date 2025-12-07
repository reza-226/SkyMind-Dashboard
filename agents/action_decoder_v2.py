import torch
import torch.nn.functional as F
import numpy as np

class ActionDecoder:
    """
    Convert actor outputs into environment actions (Version 2).
    
    Improvements over v1:
    - Better handling of edge cases
    - Support for batch and single sample decoding
    - More flexible scaling options
    - Validation of outputs
    """

    def __init__(self, max_movement_step=5.0, min_cpu=0.1, min_bandwidth=0.05):
        """
        Args:
            max_movement_step: Maximum movement distance per step
            min_cpu: Minimum CPU allocation (prevents zero allocation)
            min_bandwidth: Minimum bandwidth per channel (prevents zero allocation)
        """
        self.max_step = max_movement_step
        self.min_cpu = min_cpu
        self.min_bandwidth = min_bandwidth

    def decode(self, offload_logits, cont, deterministic=False):
        """
        Decode actor outputs into environment actions.
        
        Args:
            offload_logits: (B, 5) - logits for offload destination
            cont: (B, 6) - continuous actions in [-1, 1]
            deterministic: If True, use argmax for offload; else sample
            
        Returns:
            List of action dicts or single dict if batch size is 1
        """
        B = offload_logits.size(0)

        # Choose offload destination
        if deterministic:
            offload_choice = torch.argmax(offload_logits, dim=-1)
        else:
            # Sample from categorical distribution
            probs = F.softmax(offload_logits, dim=-1)
            offload_choice = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Split continuous actions
        cpu_raw = cont[:, 0]
        bw_raw = cont[:, 1:4]
        move_raw = cont[:, 4:6]

        # CPU allocation ∈ [min_cpu, 1.0]
        cpu = self.min_cpu + (1.0 - self.min_cpu) * (cpu_raw + 1) / 2

        # Bandwidth: normalize to probability simplex with minimum threshold
        bw_softmax = F.softmax(bw_raw, dim=-1)
        # Ensure minimum bandwidth per channel
        bw = bw_softmax * (1.0 - 3 * self.min_bandwidth) + self.min_bandwidth
        # Renormalize to ensure sum = 1
        bw = bw / bw.sum(dim=-1, keepdim=True)

        # Movement: scaled to [-max_step, +max_step]
        move = move_raw * self.max_step

        # Build action list
        actions = []
        for i in range(B):
            action = {
                "offload": int(offload_choice[i].item()),
                "cpu": float(cpu[i].item()),
                "bandwidth": bw[i].detach().cpu().numpy(),
                "move": move[i].detach().cpu().numpy()
            }
            
            # Validate action
            action = self._validate_action(action)
            actions.append(action)

        # Return single dict if batch size is 1
        return actions[0] if B == 1 else actions

    def _validate_action(self, action):
        """Validate and clip action values to valid ranges"""
        # Clip CPU to [0, 1]
        action["cpu"] = np.clip(action["cpu"], 0.0, 1.0)
        
        # Ensure bandwidth sums to 1 and all values are positive
        bw = np.array(action["bandwidth"])
        bw = np.maximum(bw, self.min_bandwidth)
        action["bandwidth"] = bw / bw.sum()
        
        # Clip movement
        action["move"] = np.clip(action["move"], -self.max_step, self.max_step)
        
        # Validate offload choice
        action["offload"] = int(np.clip(action["offload"], 0, 4))
        
        return action

    def decode_single(self, offload_logits, cont, deterministic=True):
        """
        Convenience method for decoding single sample.
        
        Args:
            offload_logits: (5,) or (1, 5)
            cont: (6,) or (1, 6)
            deterministic: If True, use argmax
            
        Returns:
            Single action dict
        """
        # Ensure batch dimension
        if offload_logits.dim() == 1:
            offload_logits = offload_logits.unsqueeze(0)
        if cont.dim() == 1:
            cont = cont.unsqueeze(0)
        
        return self.decode(offload_logits, cont, deterministic)
