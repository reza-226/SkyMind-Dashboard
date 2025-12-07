import torch
import torch.optim as optim
import numpy as np
from .actor_network_v2 import ActorNetwork
from .critic_network_v2 import CriticNetwork
from .action_decoder_v2 import ActionDecoder

class MADDPGAgent:
    """
    MADDPG Agent for UAV Offloading (Version 2).
    
    Improvements over v1:
    - Better hyperparameter management
    - Support for different network architectures
    - Improved exploration strategies
    - Better logging and debugging
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary containing all hyperparameters
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Network parameters
        state_dim = config.get('state_dim', 537)
        offload_dim = config.get('offload_dim', 5)
        continuous_dim = config.get('continuous_dim', 6)
        action_dim = config.get('action_dim', 11)
        hidden_size = config.get('hidden_size', 512)
        
        # Initialize networks
        self.actor = ActorNetwork(
            state_dim=state_dim,
            offload_dim=offload_dim,
            continuous_dim=continuous_dim,
            hidden=hidden_size,
            use_layer_norm=config.get('use_layer_norm', False),
            dropout=config.get('dropout', 0.0)
        ).to(self.device)
        
        self.critic = CriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden=hidden_size,
            use_layer_norm=config.get('use_layer_norm', False),
            num_hidden_layers=config.get('num_hidden_layers', 2)
        ).to(self.device)
        
        # Target networks
        self.actor_target = ActorNetwork(
            state_dim=state_dim,
            offload_dim=offload_dim,
            continuous_dim=continuous_dim,
            hidden=hidden_size,
            use_layer_norm=config.get('use_layer_norm', False),
            dropout=0.0  # No dropout in target network
        ).to(self.device)
        
        self.critic_target = CriticNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden=hidden_size,
            use_layer_norm=config.get('use_layer_norm', False),
            num_hidden_layers=config.get('num_hidden_layers', 2)
        ).to(self.device)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.get('actor_lr', 1e-4)
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.get('critic_lr', 1e-3)
        )
        
        # Action decoder
        self.action_decoder = ActionDecoder(
            max_movement_step=config.get('max_movement_step', 5.0),
            min_cpu=config.get('min_cpu', 0.1),
            min_bandwidth=config.get('min_bandwidth', 0.05)
        )
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)
        self.exploration_noise = config.get('exploration_noise', 0.1)
        
    def select_action(self, state, explore=True):
        """
        Select action given state.
        
        Args:
            state: State tensor or numpy array
            explore: If True, add exploration noise
            
        Returns:
            Action dictionary
        """
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            offload_logits, cont = self.actor(state)
            
            # Add exploration noise if needed
            if explore:
                noise = torch.randn_like(cont) * self.exploration_noise
                cont = torch.clamp(cont + noise, -1.0, 1.0)
        
        self.actor.train()
        
        # Decode action
        action = self.action_decoder.decode_single(
            offload_logits.squeeze(0),
            cont.squeeze(0),
            deterministic=not explore
        )
        
        return action
    
    def soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
