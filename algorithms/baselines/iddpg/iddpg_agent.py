import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .actor_network import IndependentActorNetwork
from .critic_network import IndependentCriticNetwork

class IDDPGAgent:
    """
    Independent DDPG Agent.
    هر UAV مستقل آموزش می‌بینه بدون اطلاع از agent‌های دیگر.
    
    ✅ سازگار با Real Environment (state=35, action=4)
    ✅ سازگار با Fake Environment (state=268, action=11)
    """

    def __init__(
        self,
        agent_id,
        local_state_dim=268,
        action_dim=11,
        offload_dim=5,
        continuous_dim=6,
        hidden=512,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        device="cpu",
        use_simple_action=False  # ✅ برای Real Environment
    ):
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.use_simple_action = use_simple_action
        
        # ✅ ذخیره ابعاد واقعی
        self.state_dim = local_state_dim
        self.action_dim = action_dim
        self.offload_dim = offload_dim
        self.continuous_dim = continuous_dim

        print(f"\n🔧 Initializing Agent {agent_id}:")
        print(f"   State dim: {local_state_dim}")
        print(f"   Action dim: {action_dim}")
        print(f"   Simple action mode: {use_simple_action}")

        # Actor
        self.actor = IndependentActorNetwork(
            local_state_dim, offload_dim, continuous_dim, hidden
        ).to(device)
        self.actor_target = IndependentActorNetwork(
            local_state_dim, offload_dim, continuous_dim, hidden
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic
        self.critic = IndependentCriticNetwork(
            local_state_dim, action_dim, hidden
        ).to(device)
        self.critic_target = IndependentCriticNetwork(
            local_state_dim, action_dim, hidden
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, local_state, explore=True, epsilon=0.1):
        """
        فقط با local state خودش تصمیم می‌گیره.
        """
        with torch.no_grad():
            local_state = torch.FloatTensor(local_state).unsqueeze(0).to(self.device)
            offload_logits, cont = self.actor(local_state)
            
            if explore and np.random.rand() < epsilon:
                # Random exploration
                offload = np.random.randint(0, self.offload_dim)
                cpu = np.random.rand()
                bandwidth = np.random.dirichlet([1, 1, 1])
                move = np.random.randn(2) * 5.0
            else:
                offload = torch.argmax(offload_logits, dim=-1).item()
                cpu = ((cont[0, 0] + 1) / 2).item()
                bw_raw = cont[0, 1:4]
                bandwidth = torch.softmax(bw_raw, dim=-1).cpu().numpy()
                move = (cont[0, 4:6] * 5.0).cpu().numpy()

            # ✅ اگر Real Environment بود، فرمت ساده
            if self.use_simple_action:
                return {
                    "move": move,           # (2,) dx, dy
                    "dz": 0.0,             # scalar
                    "offload": offload      # int 0-4
                }
            
            # ❌ اگر Fake Environment بود، فرمت پیچیده
            return {
                "offload": offload,
                "cpu": cpu,
                "bandwidth": bandwidth,
                "move": move
            }

    def update(self, batch):
        """
        آپدیت مستقل بدون دیدن action‌های agent‌های دیگر.
        
        batch: {
            'local_state': (B, state_dim),
            'action': (B, action_dim),
            'reward': (B, 1),
            'next_local_state': (B, state_dim),
            'done': (B, 1)
        }
        """
        local_state = batch['local_state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_local_state = batch['next_local_state'].to(self.device)
        done = batch['done'].to(self.device)

        # ✅ اطمینان از تطابق ابعاد action
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # ✅ اگر action بیشتر از action_dim داشت، کوتاهش کن
        if action.shape[-1] > self.action_dim:
            action = action[..., :self.action_dim]
        # ✅ اگر کمتر داشت، pad کن
        elif action.shape[-1] < self.action_dim:
            padding = torch.zeros(
                action.shape[0], 
                self.action_dim - action.shape[-1],
                device=self.device
            )
            action = torch.cat([action, padding], dim=-1)

        # Critic update
        with torch.no_grad():
            next_offload_logits, next_cont = self.actor_target(next_local_state)
            next_action = self._combine_action(next_offload_logits, next_cont)
            
            # ✅ همین چک رو برای next_action هم اعمال کن
            if next_action.shape[-1] > self.action_dim:
                next_action = next_action[..., :self.action_dim]
            elif next_action.shape[-1] < self.action_dim:
                padding = torch.zeros(
                    next_action.shape[0], 
                    self.action_dim - next_action.shape[-1],
                    device=self.device
                )
                next_action = torch.cat([next_action, padding], dim=-1)
            
            target_q = self.critic_target(next_local_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q = self.critic(local_state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # ✅ Gradient clipping
        self.critic_optimizer.step()

        # Actor update
        offload_logits, cont = self.actor(local_state)
        action_pred = self._combine_action(offload_logits, cont)
        
        # ✅ همین چک رو برای action_pred هم اعمال کن
        if action_pred.shape[-1] > self.action_dim:
            action_pred = action_pred[..., :self.action_dim]
        elif action_pred.shape[-1] < self.action_dim:
            padding = torch.zeros(
                action_pred.shape[0], 
                self.action_dim - action_pred.shape[-1],
                device=self.device
            )
            action_pred = torch.cat([action_pred, padding], dim=-1)
        
        actor_loss = -self.critic(local_state, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # ✅ Gradient clipping
        self.actor_optimizer.step()

        # Soft update targets
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }

    def _combine_action(self, offload_logits, cont):
        """
        Combine discrete + continuous into single action tensor.
        ✅ خروجی حداکثر به اندازه action_dim
        """
        offload_onehot = torch.zeros(offload_logits.size(0), self.offload_dim).to(self.device)
        offload_idx = torch.argmax(offload_logits, dim=-1)
        offload_onehot.scatter_(1, offload_idx.unsqueeze(1), 1.0)
        
        combined = torch.cat([offload_onehot, cont], dim=-1)
        
        # ✅ برش بزن به اندازه action_dim
        if combined.shape[-1] > self.action_dim:
            combined = combined[..., :self.action_dim]
        
        return combined

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'config': {
                'agent_id': self.agent_id,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'offload_dim': self.offload_dim,
                'continuous_dim': self.continuous_dim
            }
        }, path)
        print(f"💾 Agent {self.agent_id} saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        print(f"📂 Agent {self.agent_id} loaded from {path}")
