"""
train_maddpg_complete.py
Complete MADDPG Training Pipeline with Curriculum Learning - DIMENSION FIX
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# PettingZoo
from pettingzoo.mpe import simple_tag_v3

# Local imports
sys.path.append(str(Path(__file__).parent))
from configs.curriculum_config import TRAINING_CONFIG, CURRICULUM_STAGES
from utils.ou_noise import OUNoise

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Actor(nn.Module):
    """Actor Network"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Actions in [0, 1]
        )
        
    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """Centralized Critic Network"""
    
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)


class MADDPGAgent:
    """MADDPG Agent with Exploration Noise"""
    
    def __init__(self, obs_dim, action_dim, agent_id, config, device='cpu'):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Networks
        hidden_dim = config['hidden_dim']
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=config['lr_actor']
        )
        
        # Exploration noise
        self.noise = OUNoise(action_dim)
        self.exploration_noise = config['exploration_noise']
        
    def select_action(self, obs, add_noise=True):
        """Select action with optional exploration noise"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs_tensor).cpu().numpy()[0]
            
            if add_noise:
                noise = self.noise.sample() * self.exploration_noise
                action = np.clip(action + noise, 0, 1)
                
        return action
    
    def update_target(self, tau):
        """Soft update of target network"""
        for param, target_param in zip(
            self.actor.parameters(), 
            self.actor_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, experience):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class MADDPGTrainer:
    """MADDPG Trainer with Curriculum Learning"""
    
    def __init__(self, env, config, stage_name='Level1'):
        self.env = env
        self.config = config
        self.stage_name = stage_name
        self.device = torch.device(config['device'])
        
        # Get environment info - FIX: Use parallel API with per-agent dimensions
        obs, _ = self.env.reset()
        self.agents = list(obs.keys())
        self.num_agents = len(self.agents)
        
        # ✅ FIX: Get observation dimensions PER AGENT
        self.obs_dims = {}
        self.action_dim = None
        
        for agent_id in self.agents:
            self.obs_dims[agent_id] = self.env.observation_space(agent_id).shape[0]
            if self.action_dim is None:
                self.action_dim = self.env.action_space(agent_id).shape[0]
        
        logger.info(f"Environment: {self.num_agents} agents")
        logger.info(f"Agents: {self.agents}")
        logger.info(f"Observation dimensions per agent:")
        for agent_id, obs_dim in self.obs_dims.items():
            logger.info(f"  {agent_id}: obs_dim={obs_dim}")
        logger.info(f"Action dim (shared): {self.action_dim}")
        
        # Initialize agents with INDIVIDUAL observation dimensions
        self.maddpg_agents = {}
        for agent_id in self.agents:
            self.maddpg_agents[agent_id] = MADDPGAgent(
                self.obs_dims[agent_id],  # ✅ Per-agent dimension
                self.action_dim,
                agent_id,
                config,
                self.device
            )
        
        # Centralized critic with TOTAL observation dimension
        total_obs_dim = sum(self.obs_dims.values())
        total_action_dim = self.action_dim * self.num_agents
        
        logger.info(f"Critic input: obs_dim={total_obs_dim}, action_dim={total_action_dim}")
        
        self.critic = Critic(
            total_obs_dim, 
            total_action_dim,
            config['hidden_dim']
        ).to(self.device)
        self.critic_target = Critic(
            total_obs_dim,
            total_action_dim,
            config['hidden_dim']
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config['lr_critic']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        
        # TensorBoard
        log_dir = f"runs/{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
        
        # Metrics
        self.episode_rewards = {agent_id: [] for agent_id in self.agents}
        self.best_reward = -float('inf')
        
    def train(self, total_episodes):
        """Main training loop"""
        logger.info(f"Starting training: {total_episodes} episodes")
        
        pbar = tqdm(range(total_episodes), desc=f"Training {self.stage_name}")
        
        for episode in pbar:
            episode_reward = self.run_episode(train=True)
            
            # Decay exploration noise
            for agent in self.maddpg_agents.values():
                agent.exploration_noise = max(
                    self.config['min_noise'],
                    agent.exploration_noise * self.config['noise_decay']
                )
            
            # Logging
            if episode % self.config['log_frequency'] == 0:
                self.log_metrics(episode, episode_reward)
                
            # Evaluation
            if episode % self.config['eval_frequency'] == 0:
                eval_reward = self.evaluate(num_episodes=10)
                self.writer.add_scalar('eval/mean_reward', eval_reward, episode)
                logger.info(f"Episode {episode}: Eval reward = {eval_reward:.2f}")
                
            # Save checkpoint
            if episode % self.config['save_frequency'] == 0:
                self.save_checkpoint(episode)
                
            # Update progress bar
            mean_reward = np.mean(list(episode_reward.values()))
            pbar.set_postfix({
                'reward': f"{mean_reward:.2f}",
                'noise': f"{self.maddpg_agents[self.agents[0]].exploration_noise:.3f}"
            })
        
        # Save final model
        self.save_checkpoint('final')
        logger.info(f"Training {self.stage_name} completed!")
        
    def run_episode(self, train=True):
        """Run single episode using parallel API"""
        observations, infos = self.env.reset()
        episode_reward = {agent_id: 0 for agent_id in self.agents}
        done = False
        step = 0
        max_steps = 50  # Max steps per episode
        
        while not done and step < max_steps:
            # Select actions for all agents
            actions = {}
            for agent_id in self.agents:
                obs = observations[agent_id]
                add_noise = train
                actions[agent_id] = self.maddpg_agents[agent_id].select_action(
                    obs, add_noise
                )
            
            # Step environment
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Update episode rewards
            for agent_id in self.agents:
                episode_reward[agent_id] += rewards[agent_id]
            
            # Store experience
            if train:
                # Prepare all observations and actions (order-consistent)
                all_obs = [observations[aid] for aid in self.agents]
                all_actions = [actions[aid] for aid in self.agents]
                all_next_obs = [next_observations[aid] for aid in self.agents]
                all_rewards = [rewards[aid] for aid in self.agents]
                all_dones = [terminations[aid] or truncations[aid] for aid in self.agents]
                
                self.replay_buffer.push({
                    'obs': all_obs,
                    'actions': all_actions,
                    'rewards': all_rewards,
                    'next_obs': all_next_obs,
                    'dones': all_dones
                })
            
            # Check if episode is done
            done = all(terminations.values()) or all(truncations.values())
            
            # Update observations
            observations = next_observations
            step += 1
        
        # Update networks
        if train and len(self.replay_buffer) >= self.config['min_buffer_size']:
            for _ in range(4):  # Multiple updates per episode
                self.update_networks()
        
        return episode_reward
    
    def update_networks(self):
        """Update actor and critic networks"""
        batch = self.replay_buffer.sample(self.config['batch_size'])
        
        # ✅ FIX: Prepare batch with variable-length observations
        obs_list = []
        next_obs_list = []
        
        for exp in batch:
            # Concatenate observations from all agents
            obs_concat = np.concatenate(exp['obs'])
            next_obs_concat = np.concatenate(exp['next_obs'])
            obs_list.append(obs_concat)
            next_obs_list.append(next_obs_concat)
        
        obs_batch = torch.FloatTensor(obs_list).to(self.device)
        
        actions_batch = torch.FloatTensor([
            np.concatenate(exp['actions']) for exp in batch
        ]).to(self.device)
        
        rewards_batch = torch.FloatTensor([
            np.mean(exp['rewards']) for exp in batch
        ]).unsqueeze(1).to(self.device)
        
        next_obs_batch = torch.FloatTensor(next_obs_list).to(self.device)
        
        dones_batch = torch.FloatTensor([
            float(any(exp['dones'])) for exp in batch
        ]).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = []
            start_idx = 0
            for agent_id in self.agents:
                obs_dim = self.obs_dims[agent_id]
                next_obs_agent = next_obs_batch[:, start_idx:start_idx + obs_dim]
                next_action = self.maddpg_agents[agent_id].actor_target(next_obs_agent)
                next_actions.append(next_action)
                start_idx += obs_dim
            next_actions = torch.cat(next_actions, dim=1)
            
            target_q = self.critic_target(next_obs_batch, next_actions)
            target_q = rewards_batch + self.config['gamma'] * target_q * (1 - dones_batch)
        
        current_q = self.critic(obs_batch, actions_batch)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Update Actors
        start_idx = 0
        for i, agent_id in enumerate(self.agents):
            obs_dim = self.obs_dims[agent_id]
            obs_agent = obs_batch[:, start_idx:start_idx + obs_dim]
            
            # Compute actor loss
            actions = []
            action_start_idx = 0
            for j, aid in enumerate(self.agents):
                aid_obs_dim = self.obs_dims[aid]
                obs_other = obs_batch[:, action_start_idx:action_start_idx + aid_obs_dim]
                
                if aid == agent_id:
                    actions.append(self.maddpg_agents[aid].actor(obs_agent))
                else:
                    with torch.no_grad():
                        actions.append(self.maddpg_agents[aid].actor(obs_other))
                
                action_start_idx += aid_obs_dim
            
            actions = torch.cat(actions, dim=1)
            
            actor_loss = -self.critic(obs_batch, actions).mean()
            
            self.maddpg_agents[agent_id].actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.maddpg_agents[agent_id].actor.parameters(), 0.5
            )
            self.maddpg_agents[agent_id].actor_optimizer.step()
            
            start_idx += obs_dim
        
        # Soft update
        for agent in self.maddpg_agents.values():
            agent.update_target(self.config['tau'])
        
        # Update critic target
        for param, target_param in zip(
            self.critic.parameters(), 
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.config['tau'] * param.data + 
                (1 - self.config['tau']) * target_param.data
            )
    
    def evaluate(self, num_episodes=10):
        """Evaluate current policy"""
        total_reward = 0
        for _ in range(num_episodes):
            episode_reward = self.run_episode(train=False)
            total_reward += np.mean(list(episode_reward.values()))
        return total_reward / num_episodes
    
    def log_metrics(self, episode, episode_reward):
        """Log metrics to TensorBoard"""
        mean_reward = np.mean(list(episode_reward.values()))
        self.writer.add_scalar('train/mean_reward', mean_reward, episode)
        
        # Log per-agent rewards
        for agent_id, reward in episode_reward.items():
            self.writer.add_scalar(f'train/reward_{agent_id}', reward, episode)
        
        self.writer.add_scalar(
            'train/exploration_noise',
            self.maddpg_agents[self.agents[0]].exploration_noise,
            episode
        )
        self.writer.add_scalar('train/buffer_size', len(self.replay_buffer), episode)
    
    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        save_dir = f"models/{self.stage_name}/checkpoint_{episode}"
        os.makedirs(save_dir, exist_ok=True)
        
        for agent_id, agent in self.maddpg_agents.items():
            torch.save(
                agent.actor.state_dict(),
                f"{save_dir}/{agent_id}.pth"
            )
        
        torch.save(self.critic.state_dict(), f"{save_dir}/critic.pth")
        logger.info(f"Saved checkpoint: {save_dir}")
    
    def load_checkpoint(self, checkpoint_dir):
        """Load model checkpoint"""
        for agent_id, agent in self.maddpg_agents.items():
            checkpoint_path = f"{checkpoint_dir}/{agent_id}.pth"
            if os.path.exists(checkpoint_path):
                agent.actor.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.device)
                )
                agent.actor_target.load_state_dict(agent.actor.state_dict())
                logger.info(f"Loaded {agent_id} from {checkpoint_path}")


def create_env(env_config):
    """Create environment from config"""
    return simple_tag_v3.parallel_env(
        num_good=env_config['num_good'],
        num_adversaries=env_config['num_adversaries'],
        num_obstacles=env_config['num_obstacles'],
        max_cycles=50,
        continuous_actions=True
    )


def main():
    """Main curriculum training pipeline"""
    logger.info("="*80)
    logger.info("MADDPG Curriculum Training Pipeline")
    logger.info("="*80)
    
    prev_checkpoint = None
    
    for stage in CURRICULUM_STAGES:
        logger.info(f"\n{'='*80}")
        logger.info(f"Stage: {stage['name']}")
        logger.info(f"Description: {stage['description']}")
        logger.info(f"Episodes: {stage['episodes']}")
        logger.info(f"{'='*80}\n")
        
        # Create environment
        env = create_env(stage['env_config'])
        
        # Create trainer
        trainer = MADDPGTrainer(env, TRAINING_CONFIG, stage['name'])
        
        # Transfer learning
        if prev_checkpoint is not None:
            logger.info(f"Loading previous checkpoint: {prev_checkpoint}")
            try:
                trainer.load_checkpoint(prev_checkpoint)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Train
        start_time = time.time()
        trainer.train(stage['episodes'])
        elapsed = time.time() - start_time
        
        logger.info(f"Stage {stage['name']} completed in {elapsed/3600:.2f} hours")
        
        # Update checkpoint path
        prev_checkpoint = f"models/{stage['name']}/checkpoint_final"
        
        # Close environment
        env.close()
    
    logger.info("\n" + "="*80)
    logger.info("🎉 All training stages completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
