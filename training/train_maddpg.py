"""
MADDPG Training Script - Multi-Agent Version with Comprehensive Logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Import environment
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.multi_uav_wrapper import MultiUAVWrapper
from models.actor_critic.actor_network import ActorNetwork
from models.actor_critic.critic_network import CriticNetwork
from models.actor_critic.action_decoder import ActionDecoder
from utils.replay_buffer import ReplayBuffer
from utils.logger import TrainingLogger
from utils.evaluator import Evaluator


class MADDPGAgent:
    """Wrapper class for single agent (Actor + Critic)"""
    
    def __init__(self, actor, device):
        self.actor = actor
        self.device = device
    
    def select_action(self, state, add_noise=False, noise_scale=0.1):
        """
        Select action from actor network
        
        Args:
            state: torch.Tensor (1, state_dim)
            add_noise: bool
            noise_scale: float
            
        Returns:
            action: numpy array (action_dim,)
        """
        with torch.no_grad():
            offload_logits, _ = self.actor(state)
            
            if add_noise:
                noise = torch.randn_like(offload_logits) * noise_scale
                offload_logits = offload_logits + noise
            
            return offload_logits.cpu().numpy()[0]


class MADDPGTrainer:
    """Multi-Agent DDPG Trainer with Comprehensive Metrics"""
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Environment - Multi-agent wrapper
        self.env = MultiUAVWrapper(
            n_agents=config['n_agents'],
            num_tasks=100,
            task_complexity='mixed',
            max_steps=config['max_steps_per_episode']
        )
        
        # Networks
        self.n_agents = config['n_agents']
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        
        # Create actors and critics
        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(self.n_agents):
            # Actor
            actor = ActorNetwork(
                state_dim=self.state_dim,
                offload_dim=5,
                continuous_dim=0,
                hidden_dim=config['hidden_dim']
            ).to(self.device)
            
            actor_target = ActorNetwork(
                state_dim=self.state_dim,
                offload_dim=5,
                continuous_dim=0,
                hidden_dim=config['hidden_dim']
            ).to(self.device)
            actor_target.load_state_dict(actor.state_dict())
            
            # Critic
            critic = CriticNetwork(
                state_dim=self.state_dim,
                action_dim=5,
                hidden_dim=config['hidden_dim']
            ).to(self.device)
            
            critic_target = CriticNetwork(
                state_dim=self.state_dim,
                action_dim=5,
                hidden_dim=config['hidden_dim']
            ).to(self.device)
            critic_target.load_state_dict(critic.state_dict())
            
            # Optimizers
            actor_opt = optim.Adam(actor.parameters(), lr=config['actor_lr'])
            critic_opt = optim.Adam(critic.parameters(), lr=config['critic_lr'])
            
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.actor_optimizers.append(actor_opt)
            self.critic_optimizers.append(critic_opt)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config['buffer_size'],
            n_agents=self.n_agents
        )
        
        # Action Decoder
        self.decoder = ActionDecoder()
        
        # Training parameters
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.tau = config['tau']
        
        # Logger
        self.logger = TrainingLogger(config['checkpoint_dir'])
        
        # 🔧 Create agents dictionary for Evaluator
        self.agents_dict = {
            f'uav_{i}': MADDPGAgent(self.actors[i], self.device) 
            for i in range(self.n_agents)
        }
        
        # 🔧 Evaluator with correct signature
        self.evaluator = Evaluator(
            env=self.env,
            agents=self.agents_dict,
            n_eval_episodes=5,
            max_steps=config['max_steps_per_episode'],
            device=self.device
        )
        
        # ✅ Training History - Comprehensive Metrics
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'avg_rewards': [],
            'noise_levels': [],
            'critic_losses': [],
            'actor_losses': [],
            'eval_rewards': [],
            'eval_episodes': [],
            'timestamps': []
        }
        
        # ✅ Best Model Tracking
        self.best_reward = float('-inf')
        self.best_episode = 0
        self.recent_rewards = []  # For moving average
        
        # Episode counter
        self.episode = 0
        self.total_steps = 0
        self.start_time = time.time()
        
    def select_actions(self, observation, explore=True, epsilon=0.1):
        """
        Select actions for all agents
        
        Args:
            observation: Environment observation (dict: {agent_id: state})
            explore: Whether to add exploration noise
            epsilon: Exploration noise scale
            
        Returns:
            env_actions: Dict of actions for environment {agent_id: numpy_array(11)}
            action_vectors: Numpy array of raw network outputs for replay buffer
        """
        action_vectors = []
        
        for i in range(self.n_agents):
            agent_id = f'uav_{i}'
            state = observation[agent_id]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get network output (only offload logits now)
                offload_logits, _ = self.actors[i](state_tensor)
                
                if explore:
                    # Add exploration noise
                    noise = torch.randn_like(offload_logits) * epsilon
                    offload_logits = offload_logits + noise
            
            # Store raw network output for replay buffer
            action_vectors.append(offload_logits.cpu().numpy()[0])
        
        # Decode actions for environment
        action_vectors_np = np.array(action_vectors)
        env_actions = self.decoder.decode_batch(action_vectors_np)
        
        return env_actions, action_vectors_np
    
    def update(self):
        """Update all agents using MADDPG"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        critic_losses = []
        actor_losses = []
        
        # Update each agent
        for i in range(self.n_agents):
            # ===== Critic Update =====
            with torch.no_grad():
                # Get next actions from target actors
                next_offload_logits, _ = self.actor_targets[i](next_states[:, i])
                
                # Target Q-value
                target_q = self.critic_targets[i](next_states[:, i], next_offload_logits)
                target_q = rewards[:, i:i+1] + self.gamma * (1 - dones[:, i:i+1]) * target_q
            
            # Current Q-value
            current_q = self.critics[i](states[:, i], actions[:, i])
            
            # Critic loss
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critic_optimizers[i].step()
            
            critic_losses.append(critic_loss.item())
            
            # ===== Actor Update =====
            offload_logits, _ = self.actors[i](states[:, i])
            
            # Actor loss (maximize Q)
            actor_loss = -self.critics[i](states[:, i], offload_logits).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()
            
            actor_losses.append(actor_loss.item())
            
            # Soft update target networks
            self._soft_update(self.actors[i], self.actor_targets[i])
            self._soft_update(self.critics[i], self.critic_targets[i])
        
        return {
            'critic_loss': np.mean(critic_losses),
            'actor_loss': np.mean(actor_losses)
        }
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
    
    def train_episode(self, epsilon):
        """Train one episode"""
        observation, info = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select actions
            env_actions, action_vectors = self.select_actions(
                observation,
                explore=True,
                epsilon=epsilon
            )
            
            # Step environment
            next_observation, rewards, terminated, truncated, info = self.env.step(env_actions)
            
            # Convert observations to lists for replay buffer
            obs_list = [observation[f'uav_{i}'] for i in range(self.n_agents)]
            next_obs_list = [next_observation[f'uav_{i}'] for i in range(self.n_agents)]
            rewards_list = [rewards[f'uav_{i}'] for i in range(self.n_agents)]
            
            # Store transition
            self.replay_buffer.push(
                obs_list,
                action_vectors,
                rewards_list,
                next_obs_list,
                terminated
            )
            
            # Update
            losses = self.update()
            
            episode_reward += sum(rewards_list)
            episode_steps += 1
            self.total_steps += 1
            observation = next_observation
            done = terminated
            
            if episode_steps >= self.config['max_steps_per_episode']:
                break
        
        self.episode += 1
        
        return {
            'episode': self.episode,
            'reward': episode_reward,
            'steps': episode_steps,
            'losses': losses if losses else {'critic_loss': 0, 'actor_loss': 0}
        }
    
    def train(self, num_episodes):
        """Main training loop with comprehensive logging"""
        print("=" * 80)
        print("🚀 MADDPG Training Started")
        print("=" * 80)
        print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  Device: {self.device}")
        print(f"🤖 Number of Agents: {self.n_agents}")
        print(f"📊 Total Episodes: {num_episodes}")
        print(f"💾 Checkpoint Directory: {self.config['checkpoint_dir']}")
        print("=" * 80)
        
        for ep in range(num_episodes):
            # ✅ Dynamic epsilon (exploration noise decay)
            epsilon = max(0.01, 1.0 - ep / (num_episodes * 0.7))
            
            # Train one episode
            result = self.train_episode(epsilon)
            
            # ✅ Update recent rewards for moving average
            self.recent_rewards.append(result['reward'])
            if len(self.recent_rewards) > 100:
                self.recent_rewards.pop(0)
            
            avg_reward_100 = np.mean(self.recent_rewards)
            
            # ✅ Store metrics in history
            self.training_history['episodes'].append(result['episode'])
            self.training_history['rewards'].append(result['reward'])
            self.training_history['steps'].append(result['steps'])
            self.training_history['avg_rewards'].append(avg_reward_100)
            self.training_history['noise_levels'].append(epsilon)
            self.training_history['critic_losses'].append(result['losses']['critic_loss'])
            self.training_history['actor_losses'].append(result['losses']['actor_loss'])
            self.training_history['timestamps'].append(time.time() - self.start_time)
            
            # ✅ Log to TrainingLogger
            self.logger.log_episode(result)
            
            # ✅ Check for best model
            if result['reward'] > self.best_reward:
                self.best_reward = result['reward']
                self.best_episode = result['episode']
                self.save_checkpoint('best_model.pt')
                print(f"🎉 New best reward: {self.best_reward:.2f} at episode {self.best_episode}")
            
            # ✅ Print progress (every 10 episodes)
            if (ep + 1) % self.config['log_interval'] == 0:
                elapsed = time.time() - self.start_time
                print(f"📊 Episode {result['episode']}/{num_episodes} | "
                      f"Steps: {result['steps']} | "
                      f"Reward: {result['reward']:.2f} | "
                      f"Avg(100): {avg_reward_100:.2f} | "
                      f"Noise: {epsilon:.4f} | "
                      f"Time: {str(timedelta(seconds=int(elapsed)))}")
            
            # ✅ Evaluate (every eval_interval episodes)
            if (ep + 1) % self.config['eval_interval'] == 0:
                print(f"\n{'='*60}")
                print(f"🔍 Running Evaluation at Episode {result['episode']}...")
                eval_result = self.evaluator.evaluate()
                self.logger.log_evaluation(eval_result)
                
                # Store eval results
                self.training_history['eval_rewards'].append(eval_result['avg_reward'])
                self.training_history['eval_episodes'].append(result['episode'])
                
                print(f"📈 Evaluation Results:")
                print(f"   - Avg Reward: {eval_result['avg_reward']:.2f}")
                print(f"   - Std Reward: {eval_result.get('std_reward', 0):.2f}")
                print(f"   - Min Reward: {eval_result.get('min_reward', 0):.2f}")
                print(f"   - Max Reward: {eval_result.get('max_reward', 0):.2f}")
                print(f"{'='*60}\n")
            
            # ✅ Save checkpoint (every save_interval episodes)
            if (ep + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f"checkpoint_episode_{result['episode']}.pt")
                print(f"💾 Checkpoint saved at episode {result['episode']}")
        
        # ✅ Training Complete
        total_time = time.time() - self.start_time
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"📊 Training Summary:")
        print(f"   - Total Episodes: {num_episodes}")
        print(f"   - Total Steps: {self.total_steps}")
        print(f"   - Best Episode: {self.best_episode}")
        print(f"   - Best Reward: {self.best_reward:.2f}")
        print(f"   - Final Avg (last 100): {avg_reward_100:.2f}")
        print(f"   - Final Noise Level: {epsilon:.4f}")
        print(f"   - Total Training Time: {str(timedelta(seconds=int(total_time)))}")
        print(f"   - Episodes per Hour: {(num_episodes / total_time * 3600):.1f}")
        print("=" * 80)
        
        # ✅ Save final results
        self.save_training_results()
        self.logger.save_summary()
        
        print(f"📁 Results saved to: {self.config['checkpoint_dir']}")
        print("=" * 80)
    
    def save_checkpoint(self, filename):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], filename)
        
        checkpoint = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers],
            'config': self.config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def save_training_results(self):
        """✅ Save comprehensive training results to JSON"""
        results = {
            'config': self.config,
            'training_history': self.training_history,
            'summary': {
                'total_episodes': self.config['num_episodes'],
                'total_steps': self.total_steps,
                'best_episode': self.best_episode,
                'best_reward': float(self.best_reward),
                'final_avg_100': float(np.mean(self.recent_rewards)) if self.recent_rewards else 0,
                'final_noise': float(self.training_history['noise_levels'][-1]) if self.training_history['noise_levels'] else 0,
                'training_time_seconds': int(time.time() - self.start_time),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Save to JSON
        results_path = os.path.join(self.config['checkpoint_dir'], 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Training results saved: {results_path}")
        
        # ✅ Also save summary only
        summary_path = os.path.join(self.config['checkpoint_dir'], 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results['summary'], f, indent=2)


def main():
    """Main function"""
    config = {
        # Environment
        'n_agents': 5,
        'state_dim': 537,
        'action_dim': 5,
        
        # Network
        'hidden_dim': 512,
        
        # Training
        'num_episodes': 1000,
        'max_steps_per_episode': 200,
        'batch_size': 256,
        'buffer_size': 100000,
        
        # Learning rates
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        
        # RL parameters
        'gamma': 0.99,
        'tau': 0.01,
        
        # Logging
        'log_interval': 10,      # ✅ Print every 10 episodes
        'eval_interval': 50,
        'save_interval': 100,
        'checkpoint_dir': 'checkpoints/maddpg'
    }
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # ✅ Save config with timestamp
    config_path = os.path.join(config['checkpoint_dir'], 'training_config.json')
    config_with_meta = {
        **config,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_with_meta, f, indent=2)
    
    print(f"📝 Config saved: {config_path}")
    
    # Train
    trainer = MADDPGTrainer(config)
    
    try:
        trainer.train(config['num_episodes'])
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠️  Training interrupted by user!")
        print("=" * 80)
        trainer.save_training_results()
        print("💾 Progress saved before exit.")
        print("=" * 80)


if __name__ == "__main__":
    main()
