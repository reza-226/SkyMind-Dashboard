"""
core/drl_agent.py - اصلاح بخش forward در ActorCritic
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from models.gnn.task_encoder import GNNTaskEncoder


class ActorCritic(nn.Module):
    """شبکه Actor-Critic با GNN Encoder"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gnn_config: Dict,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # GNN برای encode کردن task graph
        self.gnn_encoder = GNNTaskEncoder(**gnn_config)
        
        # محاسبه ابعاد ورودی شبکه‌های Actor/Critic
        # graph_embedding_dim + env_state_dim
        self.total_state_dim = gnn_config['embedding_dim'] + state_dim
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.total_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.total_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        pyg_data,
        env_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pyg_data: داده گراف PyG
            env_state: حالت محیط - shape: (batch_size, state_dim) یا (state_dim,)
        
        Returns:
            action_probs: احتمالات action
            value: مقدار state
            graph_embedding: embedding گراف
            task_embeddings: embedding های وظایف
        """
        # Encode کردن گراف
        graph_embedding, task_embeddings = self.gnn_encoder(pyg_data)
        
        # ✅ تبدیل env_state به 2D اگر 1D است
        if env_state.dim() == 1:
            env_state = env_state.unsqueeze(0)  # (state_dim,) -> (1, state_dim)
        
        # ترکیب graph embedding با env state
        full_state = torch.cat([graph_embedding, env_state], dim=-1)
        
        # محاسبه policy و value
        action_probs = self.actor(full_state)
        value = self.critic(full_state)
        
        return action_probs, value, graph_embedding, task_embeddings


class MATOAgent:
    """عامل DRL برای سیستم MATO"""
    
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        gnn_config: Dict,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        
        # ساخت شبکه Actor-Critic
        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            gnn_config=gnn_config,
            hidden_dim=hidden_dim
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # بافرهای ذخیره تجربه
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        
        # آمار
        self.total_steps = 0
        self.episode_count = 0
    
    def select_action(self, dag, env_state):
        """
        انتخاب action بر اساس DAG و env_state
        
        Args:
            dag: TaskDAG object
            env_state: numpy array با shape (state_dim,)
        
        Returns:
            action: int
            info: dict با اطلاعات اضافی
        """
        from utils.graph_utils import convert_dag_to_pyg_data
        
        # تبدیل DAG به PyG data
        pyg_data = convert_dag_to_pyg_data(dag).to(self.device)
        
        # تبدیل env_state به tensor
        env_state_tensor = torch.FloatTensor(env_state).to(self.device)
        # ✅ نیازی به unsqueeze نیست - در forward خودکار انجام می‌شود
        
        # Forward pass
        with torch.no_grad():
            action_probs, value, graph_emb, task_embs = self.policy(
                pyg_data, 
                env_state_tensor
            )
        
        # نمونه‌برداری از action
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        
        # ذخیره برای update بعدی
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        self.entropies.append(dist.entropy())
        
        self.total_steps += 1
        
        # اطلاعات اضافی
        info = {
            'action_probs': action_probs.cpu().numpy(),
            'value': value.item(),
            'graph_embedding_norm': torch.norm(graph_emb).item(),
            'entropy': dist.entropy().item()
        }
        
        return action.item(), info
    
    def store_reward(self, reward: float):
        """ذخیره reward"""
        self.rewards.append(reward)
    
    def update(self) -> Dict[str, float]:
        """
        به‌روزرسانی policy با الگوریتم A2C
        
        Returns:
            losses: dict شامل actor_loss و critic_loss
        """
        if len(self.rewards) == 0:
            return {}
        
        # محاسبه returns (با discount)
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, device=self.device)
        
        # نرمال‌سازی returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # محاسبه loss
        policy_losses = []
        value_losses = []
        
        for log_prob, value, R, entropy in zip(
            self.log_probs, self.values, returns, self.entropies
        ):
            advantage = R - value.item()
            
            # Actor loss (policy gradient)
            policy_losses.append(-log_prob * advantage - 0.01 * entropy)
            
            # Critic loss (MSE)
            value_losses.append(F.mse_loss(value, torch.tensor([[R]], device=self.device)))
        
        # ترکیب losses
        actor_loss = torch.stack(policy_losses).mean()
        critic_loss = torch.stack(value_losses).mean()
        total_loss = actor_loss + 0.5 * critic_loss
        
        # به‌روزرسانی
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        # پاک کردن بافرها
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def reset_episode(self):
        """ریست کردن برای اپیزود جدید"""
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.episode_count += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """دریافت metrics فعلی"""
        return {
            'total_steps': self.total_steps,
            'episode_count': self.episode_count
        }
    
    def save(self, path: str):
        """ذخیره مدل"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episode_count': self.episode_count
        }, path)
    
    def load(self, path: str):
        """بارگذاری مدل"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
