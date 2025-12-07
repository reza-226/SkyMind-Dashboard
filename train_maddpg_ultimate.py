# train_maddpg_ultimate.py
import os
import sys
import json
import logging
from datetime import datetime
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Compatibility patches
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

def setup_logging(log_file):
    """Setup logging"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# Neural Network Architectures
# ============================================================================

class Actor(nn.Module):
    """Actor Network - Output range [0, 1] using Sigmoid"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(self, obs):
        return self.network(obs)

class Critic(nn.Module):
    """Critic Network"""
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return self.network(x)

# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, obs, actions, rewards, next_obs, dones):
        self.buffer.append((obs, actions, rewards, next_obs, dones))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        return (
            {k: torch.FloatTensor(np.array([o[k] for o in obs])) for k in obs[0].keys()},
            {k: torch.FloatTensor(np.array([a[k] for a in actions])) for k in actions[0].keys()},
            {k: torch.FloatTensor(np.array([r[k] for r in rewards])) for k in rewards[0].keys()},
            {k: torch.FloatTensor(np.array([o[k] for o in next_obs])) for k in next_obs[0].keys()},
            {k: torch.FloatTensor(np.array([d[k] for d in dones])) for k in dones[0].keys()}
        )
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# MADDPG Agent
# ============================================================================

class MADDPGAgent:
    """MADDPG Agent"""
    def __init__(self, agent_name, obs_dim, action_dim, total_obs_dim, total_action_dim,
                 hidden_dim=128, lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01, device='cpu'):
        
        self.agent_name = agent_name
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Actor networks
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks
        self.critic = Critic(total_obs_dim, total_action_dim, hidden_dim).to(device)
        self.critic_target = Critic(total_obs_dim, total_action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def select_action(self, obs, noise_std=0.0):
        """Select action with optional noise - output in [0, 1]"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = action + noise
        
        return np.clip(action, 0, 1)
    
    def update(self, batch, agents, agent_idx):
        """Update agent networks"""
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch
        
        # Prepare data for this agent
        obs = obs_batch[self.agent_name].to(self.device)
        actions = action_batch[self.agent_name].to(self.device)
        rewards = reward_batch[self.agent_name].to(self.device)
        next_obs = next_obs_batch[self.agent_name].to(self.device)
        dones = done_batch[self.agent_name].to(self.device)
        
        # Prepare global observations and actions
        all_obs = torch.cat([obs_batch[agent.agent_name].to(self.device) for agent in agents], dim=1)
        all_actions = torch.cat([action_batch[agent.agent_name].to(self.device) for agent in agents], dim=1)
        all_next_obs = torch.cat([next_obs_batch[agent.agent_name].to(self.device) for agent in agents], dim=1)
        
        # Update Critic
        with torch.no_grad():
            next_actions = torch.cat([
                agent.actor_target(next_obs_batch[agent.agent_name].to(self.device))
                for agent in agents
            ], dim=1)
            
            target_q = self.critic_target(all_next_obs, next_actions)
            target_value = rewards.unsqueeze(1) + self.gamma * target_q * (1 - dones.unsqueeze(1))
        
        current_q = self.critic(all_obs, all_actions)
        critic_loss = nn.MSELoss()(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Update Actor
        self.actor_optimizer.zero_grad()
        
        current_actions = []
        for i, agent in enumerate(agents):
            if i == agent_idx:
                current_actions.append(self.actor(obs))
            else:
                current_actions.append(action_batch[agent.agent_name].to(self.device).detach())
        
        all_current_actions = torch.cat(current_actions, dim=1)
        actor_loss = -self.critic(all_obs, all_current_actions).mean()
        
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, path):
        """Save agent"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, os.path.join(path, f'{self.agent_name}.pth'))
    
    def load(self, path):
        """Load agent"""
        checkpoint = torch.load(os.path.join(path, f'{self.agent_name}.pth'), map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

# ============================================================================
# Training Function
# ============================================================================

def train_maddpg(
    env_name='simple_tag_v3',
    env_kwargs=None,
    max_episodes=2000,  # ✅ افزایش یافت: 1500 → 2000
    batch_size=128,  # ✅ افزایش یافت: 64 → 128
    buffer_size=150000,  # ✅ افزایش یافت: 100k → 150k
    lr_actor=1e-4,
    lr_critic=5e-4,  # ✅ کاهش یافت: 0.001 → 0.0005
    gamma=0.95,
    tau=0.01,
    noise_std=0.5,
    noise_decay=0.995,
    min_noise=0.01,
    hidden_dim=128,
    update_freq=5,
    updates_per_step=5,
    min_buffer_size=1000,
    save_interval=50,
    log_interval=10,
    model_dir='models/maddpg',
    log_file='logs/training.log',
    load_pretrained=None,
    device='cpu'
):
    """
    Train MADDPG on PettingZoo environment with OPTIMIZED PARAMETERS
    
    ✅ تغییرات اعمال‌شده:
    - max_episodes: 800 → 1500 (یادگیری بیشتر)
    - batch_size: 64 → 128 (گرادیان‌های پایدارتر)
    - lr_critic: 0.001 → 0.0005 (پایداری بیشتر)
    - buffer_size: 100k → 150k (تجربیات بیشتر)
    """
    
    # Setup logging
    logger = setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("[MADDPG] Training Started - OPTIMIZED VERSION")
    logger.info("="*80)
    logger.info(f"Environment: {env_name}")
    logger.info(f"Max episodes: {max_episodes} ✅ (افزایش‌یافته)")
    logger.info(f"Batch size: {batch_size} ✅ (افزایش‌یافته)")
    logger.info(f"Buffer size: {buffer_size} ✅ (افزایش‌یافته)")
    logger.info(f"LR Critic: {lr_critic} ✅ (کاهش‌یافته)")
    logger.info(f"LR Actor: {lr_actor}")
    logger.info(f"Gamma: {gamma}, Tau: {tau}")
    logger.info(f"Pretrained model: {load_pretrained if load_pretrained else 'None (training from scratch)'}")
    logger.info("="*80)
    
    # Load environment
    if env_kwargs is None:
        env_kwargs = {}
    
    # ✅ Enable continuous actions for MPE environments
    env_kwargs['continuous_actions'] = True
    
    try:
        if env_name == 'simple_tag_v3':
            from pettingzoo.mpe import simple_tag_v3
            env = simple_tag_v3.env(**env_kwargs)
        elif env_name == 'simple_spread_v3':
            from pettingzoo.mpe import simple_spread_v3
            env = simple_spread_v3.env(**env_kwargs)
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        
        logger.info(f"✓ Environment loaded: {env_name}")
    except Exception as e:
        logger.error(f"✗ Failed to load environment: {e}")
        raise
    
    # Get environment info
    env.reset()
    agent_names = env.agents
    
    # ✅ Get observation dimensions for EACH agent separately
    agent_obs_dims = {}
    for agent_name in agent_names:
        obs_space = env.observation_space(agent_name)
        obs_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else obs_space.n
        agent_obs_dims[agent_name] = obs_dim
    
    # Get action dimension (same for all agents in MPE)
    first_agent = agent_names[0]
    action_space = env.action_space(first_agent)
    
    if hasattr(action_space, 'shape'):
        action_dim = action_space.shape[0] if len(action_space.shape) > 0 else 1
    elif hasattr(action_space, 'n'):
        action_dim = action_space.n
    else:
        raise ValueError(f"Unknown action space type: {type(action_space)}")
    
    num_agents = len(agent_names)
    total_obs_dim = sum(agent_obs_dims.values())  # ✅ Sum of all agent obs dims
    total_action_dim = action_dim * num_agents
    
    logger.info(f"✓ Agents detected: {agent_names}")
    logger.info(f"✓ Obs dims per agent: {agent_obs_dims}")
    logger.info(f"✓ Action dim: {action_dim}")
    logger.info(f"✓ Total obs dim: {total_obs_dim}, Total action dim: {total_action_dim}")
    
    # Create agents with INDIVIDUAL observation dimensions
    agents = []
    for agent_name in agent_names:
        agent = MADDPGAgent(
            agent_name=agent_name,
            obs_dim=agent_obs_dims[agent_name],  # ✅ Use individual obs_dim
            action_dim=action_dim,
            total_obs_dim=total_obs_dim,
            total_action_dim=total_action_dim,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,  # ✅ استفاده از مقدار جدید (کمتر)
            gamma=gamma,
            tau=tau,
            device=device
        )
        agents.append(agent)
    
    logger.info(f"✓ Created {len(agents)} MADDPG agents")
    
    # Load pretrained models if specified
    if load_pretrained and os.path.exists(load_pretrained):
        logger.info(f"📥 Loading pretrained models from: {load_pretrained}")
        try:
            for agent in agents:
                agent.load(load_pretrained)
            logger.info("✓ Pretrained models loaded successfully")
        except Exception as e:
            logger.warning(f"⚠ Failed to load pretrained models: {e}")
    
    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size)
    
    # Training metrics
    episode_rewards = []
    best_reward = float('-inf')
    best_model_dir = None
    current_noise = noise_std
    
    training_history = {
        'episode': [],
        'reward': [],
        'critic_loss': [],
        'noise_std': [],
        'buffer_size': []
    }
    
    # Training loop
    logger.info("="*80)
    logger.info("[TRAIN] Starting training loop with OPTIMIZED PARAMETERS...")
    logger.info("="*80)
    
    for episode in range(max_episodes):
        env.reset()
        episode_reward = {agent: 0 for agent in agent_names}
        actor_losses = []
        critic_losses = []
        
        # Collect all observations and actions for storage
        obs_dict = {}
        action_dict = {}
        
        # Episode loop
        for agent_name in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation
            
            if done:
                action = None
            else:
                # Select action
                agent_idx = agent_names.index(agent_name)
                action = agents[agent_idx].select_action(obs, noise_std=current_noise)
                
                # Store obs and action for this agent
                obs_dict[agent_name] = obs.copy()
                action_dict[agent_name] = action.copy()
            
            env.step(action)
            episode_reward[agent_name] += reward
            
            # Store transition after all agents have acted
            if agent_name == agent_names[-1] and not done:
                # Get next observations
                next_obs_dict = {}
                for ag_name in agent_names:
                    next_obs_dict[ag_name] = env.observe(ag_name)
                
                # Create reward and done dicts
                reward_dict = {ag_name: episode_reward[ag_name] for ag_name in agent_names}
                done_dict = {ag_name: False for ag_name in agent_names}
                
                replay_buffer.add(obs_dict, action_dict, reward_dict, next_obs_dict, done_dict)
        
        # Update agents
        if len(replay_buffer) >= min_buffer_size and episode % update_freq == 0:
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(batch_size)
                for agent_idx, agent in enumerate(agents):
                    critic_loss, actor_loss = agent.update(batch, agents, agent_idx)
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
        
        # Decay noise
        current_noise = max(min_noise, current_noise * noise_decay)
        
        # Log episode
        avg_reward = np.mean(list(episode_reward.values()))
        episode_rewards.append(avg_reward)
        
        training_history['episode'].append(episode + 1)
        training_history['reward'].append(avg_reward)
        training_history['critic_loss'].append(np.mean(critic_losses) if critic_losses else 0)
        training_history['noise_std'].append(current_noise)
        training_history['buffer_size'].append(len(replay_buffer))
        
        if (episode + 1) % log_interval == 0:
            avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
            logger.info(f"Episode {episode+1}/{max_episodes} | "
                       f"Avg Reward: {avg_reward:.2f} | "
                       f"Critic Loss: {avg_critic_loss:.2f} | "
                       f"Noise: {current_noise:.4f} | "
                       f"Buffer: {len(replay_buffer)}")
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model_dir = os.path.join(model_dir, f'best_ep{episode+1}')
            os.makedirs(best_model_dir, exist_ok=True)
            for agent in agents:
                agent.save(best_model_dir)
            logger.info(f"✓ New best model saved! Reward: {best_reward:.2f} at episode {episode+1}")
        
        # Periodic save
        if (episode + 1) % save_interval == 0:
            checkpoint_dir = os.path.join(model_dir, f'checkpoint_ep{episode+1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            for agent in agents:
                agent.save(checkpoint_dir)
            logger.info(f"✓ Checkpoint saved at episode {episode+1}")
    
    env.close()
    
    # Save final model
    final_model_dir = os.path.join(model_dir, 'final')
    os.makedirs(final_model_dir, exist_ok=True)
    for agent in agents:
        agent.save(final_model_dir)
    
    # Save training history to JSON with Dashboard format
    history_path = os.path.join(model_dir, 'training_history.json')
    
    # تبدیل به فرمت Dashboard (episode به عنوان key)
    formatted_history = {}
    for i in range(len(training_history['episode'])):
        ep_num = str(training_history['episode'][i])
        formatted_history[ep_num] = {
            'episode': training_history['episode'][i],
            'avg_reward': training_history['reward'][i],
            'critic_loss': training_history['critic_loss'][i],
            'noise_std': training_history['noise_std'][i],
            'buffer_size': training_history['buffer_size'][i]
        }
    
    with open(history_path, 'w') as f:
        json.dump(formatted_history, f, indent=2)
    
    logger.info(f"✓ Training history saved: {history_path}")
    logger.info(f"✓ Total episodes saved: {len(formatted_history)}")
    
    logger.info("="*80)
    logger.info("[COMPLETE] Training finished! 🎉")
    logger.info(f"✓ Total episodes: {max_episodes}")
    logger.info(f"✓ Best reward: {best_reward:.2f}")
    logger.info(f"✓ Final model saved to: {final_model_dir}")
    logger.info(f"✓ Training history saved to: {history_path}")
    logger.info("="*80)
    
    # Return results
    return {
        'training_history': training_history,
        'best_reward': best_reward,
        'best_model_dir': best_model_dir,
        'final_model_dir': final_model_dir,
        'model_dir': model_dir,
        'total_episodes': max_episodes
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    logger = setup_logging('logs/training.log')
    
    logger.info("="*80)
    logger.info("[START] MADDPG Training - Optimized Parameters")
    logger.info("="*80)
    
    results = train_maddpg()
    
    print("\n" + "="*80)
    print("✅ Training completed successfully!")
    print("="*80)
    print(f"📊 Best reward achieved: {results['best_reward']:.2f}")
    print(f"📈 Total episodes trained: {results['total_episodes']}")
    print(f"📁 Models saved to: {results['model_dir']}")
    print(f"📄 Training history: {os.path.join(results['model_dir'], 'training_history.json')}")
    print("="*80)
