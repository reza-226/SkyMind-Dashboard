import torch
import torch.nn.functional as F

class ActionDecoder:
    """
    Convert actor outputs into environment actions.
    """

    def __init__(self, max_movement_step=5.0):
        self.max_step = max_movement_step

    def decode(self, offload_logits, cont):
        """
        Input:
            offload_logits: (B,5)
            cont: (B,6) → [-1,1]

        Output dict:
            {
              'offload': int,
              'cpu': float [0,1],
              'bandwidth': (3,) normalized sum=1,
              'move': (2,) Δx, Δy
            }
        """

        B = offload_logits.size(0)

        # choose offload destination
        offload_choice = torch.argmax(offload_logits, dim=-1)

        # continuous split
        cpu_raw = cont[:, 0]
        bw_raw = cont[:, 1:4]
        move_raw = cont[:, 4:6]

        # cpu ∈ [0,1]
        cpu = (cpu_raw + 1) / 2

        # bandwidth normalize to probability simplex
        bw = F.softmax(bw_raw, dim=-1)

        # move scaled
        move = move_raw * self.max_step

        actions = []
        for i in range(B):
            actions.append({
                "offload": int(offload_choice[i].item()),
                "cpu": float(cpu[i].item()),
                "bandwidth": bw[i].detach().cpu().numpy(),
                "move": move[i].detach().cpu().numpy()
            })

        return actions
