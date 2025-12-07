"""
train_maddpg_reward_shaped_v3_final.py
آموزش MADDPG با Reward Shaping، Dashboard Integration و Early Stopping Monitor
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
import time
import numpy as np
from pathlib import Path    
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
from tqdm import tqdm

# Add project path
project_root = Path('/content/drive/MyDrive/uav_mec')
sys.path.append(str(project_root))

# Import modules
from pettingzoo_env_maddpg import MADDPGEnv
from maddpg_agent import MADDPGAgent
from early_stopping_monitor import EarlyStoppingMonitor
from system_monitor import ThresholdConfig, SystemMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class EnhancedDashboardConnector:
    """نسخه پیشرفته Dashboard Connector با محاسبات کامل"""
    
    def __init__(self, save_dir: Path, logger=None, agent_names=None):
        """
        Initialize dashboard connector
        
        Args:
            save_dir: Directory to save dashboard data
            logger: Logger instance
            agent_names: List of agent names to track
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # Directories
        self.json_dir = self.save_dir / 'json_logs'
        self.csv_dir = self.save_dir / 'csv_exports'
        self.json_dir.mkdir(exist_ok=True)
        self.csv_dir.mkdir(exist_ok=True)
        
        # Initialize agent_data for all agents
        if agent_names is None:
            agent_names = ['agent_0', 'adversary_0', 'adversary_1']
        
        self.agent_data = {agent_name: [] for agent_name in agent_names}
        self.episode_data = []
        
        # Previous weights for drift calculation
        self.prev_weights = None
        
        if self.logger:
            self.logger.info(f"✅ EnhancedDashboardConnector initialized at {save_dir}")
            self.logger.info(f"   Tracking agents: {list(self.agent_data.keys())}")
    
    def calculate_action_saturation(self, actions: np.ndarray) -> float:
        """محاسبه نرخ اشباع اکشن‌ها"""
        threshold = 0.05
        saturated = np.sum((actions < threshold) | (actions > (1 - threshold)))
        total = actions.size
        return (saturated / total) * 100 if total > 0 else 0
    
    def calculate_weight_drift(self, agent: MADDPGAgent) -> Dict[str, float]:
        """محاسبه تغییرات وزن‌ها"""
        
        current_weights = {
            'actor': self._get_weights_norm(agent.actor),
            'critic': self._get_weights_norm(agent.critic)
        }
        
        if self.prev_weights is None:
            self.prev_weights = current_weights
            return {'actor': 0.0, 'critic': 0.0}
        
        drift = {
            'actor': abs(current_weights['actor'] - self.prev_weights['actor']),
            'critic': abs(current_weights['critic'] - self.prev_weights['critic'])
        }
        
        self.prev_weights = current_weights
        return drift
    
    def _get_weights_norm(self, network) -> float:
        """محاسبه نرم وزن‌های شبکه"""
        total_norm = 0.0
        for param in network.parameters():
            total_norm += torch.norm(param.data).item() ** 2
        return np.sqrt(total_norm)
    
    def calculate_agent_distance(self, obs: Dict) -> float:
        """محاسبه فاصله میانگین Agent-Adversary"""
        distances = []
        for i in range(3):
            agent_pos = obs[f'agent_{i}'][:2]
            for j in range(2):
                adv_pos = obs[f'adversary_{j}'][:2]
                dist = np.linalg.norm(agent_pos - adv_pos)
                distances.append(dist)
        return float(np.mean(distances)) if distances else 0.0
    
    def update_live(self, episode: int, metrics: Dict):
        """به‌روزرسانی داده‌های Live"""
        
        live_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'reward': float(metrics.get('reward', 0)),
            'critic_loss': float(metrics.get('critic_loss', 0)),
            'actor_loss': float(metrics.get('actor_loss', 0)),
            'success_rate': float(metrics.get('escape_success', 0)) * 100,
            'saturation_rate': float(metrics.get('saturation_rate', 0)),
            'epsilon': float(metrics.get('epsilon', 0))
        }
        
        # Save JSON
        with open(self.json_dir / 'live_metrics.json', 'w') as f:
            json.dump(live_data, f, indent=2)
        
        self.episode_data.append(live_data)
    
    def update_overview(self, episode: int, metrics: Dict):
        """به‌روزرسانی Overview"""
        
        overview = {
            'training_progress': {
                'current_episode': episode,
                'total_episodes': metrics.get('total_episodes', 0),
                'progress_percent': (episode / metrics.get('total_episodes', 1)) * 100
            },
            'performance_summary': {
                'avg_reward_100': float(metrics.get('avg_reward', 0)),
                'avg_success_100': float(metrics.get('avg_success', 0)) * 100,
                'best_reward': float(metrics.get('best_reward', 0)),
                'total_steps': int(metrics.get('total_steps', 0))
            },
            'network_health': {
                'critic_loss': float(metrics.get('critic_loss', 0)),
                'actor_loss': float(metrics.get('actor_loss', 0)),
                'saturation_rate': float(metrics.get('saturation_rate', 0)),
                'weight_drift_critic': float(metrics.get('drift_critic', 0))
            }
        }
        
        with open(self.json_dir / 'overview.json', 'w') as f:
            json.dump(overview, f, indent=2)
    
    def update_agents(self, episode: int, agent_metrics: Dict):
        """به‌روزرسانی داده‌های Agents"""
        
        for agent_id, metrics in agent_metrics.items():
            agent_record = {
                'episode': episode,
                'reward': float(metrics.get('reward', 0)),
                'actor_loss': float(metrics.get('actor_loss', 0)),
                'critic_loss': float(metrics.get('critic_loss', 0)),
                'actions_mean': float(metrics.get('actions_mean', 0)),
                'actions_std': float(metrics.get('actions_std', 0))
            }
            self.agent_data[agent_id].append(agent_record)
        
        # Save
        with open(self.json_dir / 'agents_data.json', 'w') as f:
            json.dump(self.agent_data, f, indent=2)
    
    def export_csv(self, episode: int):
        """خروجی CSV"""
        
        if not self.episode_data:
            return
        
        import pandas as pd
        
        df = pd.DataFrame(self.episode_data)
        csv_path = self.csv_dir / f'training_ep{episode}.csv'
        df.to_csv(csv_path, index=False)
        
        if self.logger:
            self.logger.info(f"📊 CSV exported: {csv_path}")
    
    def get_latest_metrics(self) -> Dict:
        """دریافت آخرین Metrics"""
        if self.episode_data:
            return self.episode_data[-1]
        return {}


def create_level_config(level: int) -> Dict:
    """ایجاد تنظیمات برای هر Level"""
    
    configs = {
        1: {
            'episodes': 5000,
            'n_agents': 3,
            'n_adversaries': 2,
            'max_cycles': 50,
            'buffer_size': 50000,
            'batch_size': 256,
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'gamma': 0.95,
            'tau': 0.01,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995,
            'update_every': 4,
            'warmup_episodes': 100,
            'monitor': ThresholdConfig(
               metric_name='critic_loss',
               upper_bound=8.0,
               lower_bound=None,
               window_size=100,
               check_interval=100,
               patience=5,
               min_episodes=300
            )
        },
        2: {
            'episodes': 7000,
            'n_agents': 3,
            'n_adversaries': 2,
            'max_cycles': 75,
            'buffer_size': 100000,
            'batch_size': 512,
            'lr_actor': 8e-5,
            'lr_critic': 8e-4,
            'gamma': 0.97,
            'tau': 0.008,
            'epsilon_start': 0.8,
            'epsilon_end': 0.03,
            'epsilon_decay': 0.997,
            'update_every': 3,
            'warmup_episodes': 150,
            'monitor': ThresholdConfig(
                metric_name='critic_loss',
                upper_bound=10.0,
                lower_bound=None,
                window_size=100,
                check_interval=50,
                patience=5,
                min_episodes=200
            )
        },
        3: {
            'episodes': 10000,
            'n_agents': 3,
            'n_adversaries': 2,
            'max_cycles': 100,
            'buffer_size': 150000,
            'batch_size': 1024,
            'lr_actor': 5e-5,
            'lr_critic': 5e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'epsilon_start': 0.5,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.998,
            'update_every': 2,
            'warmup_episodes': 200,
            'monitor': ThresholdConfig(
                metric_name='critic_loss',
                upper_bound=12.0,
                lower_bound=None,
                window_size=100,
                check_interval=30,
                patience=5,
                min_episodes=150
            )
        }
    }
    
    return configs.get(level, configs[2])


def train_level(level: int, load_checkpoint: Optional[str] = None):
    """
    آموزش برای یک Level مشخص
    
    Args:
        level: شماره Level (1, 2, 3)
        load_checkpoint: مسیر Checkpoint برای ادامه آموزش
    """
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 Starting Level {level} Training")
    logger.info(f"{'='*60}\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load config
    config = create_level_config(level)
    
    # Create environment
    env = MADDPGEnv(
        n_agents=config['n_agents'],
        n_adversaries=config['n_adversaries'],
        max_cycles=config['max_cycles']
    )
    
    # Initialize agents
    agents = {}
    for agent_idx, agent_name in enumerate(env.agents):
        state_dim = env.observation_spaces[agent_name].shape[0]
        action_dim = env.action_spaces[agent_name].shape[0]
        
        # Calculate total dimensions
        total_state_dim = sum(
            env.observation_spaces[a].shape[0] for a in env.agents
        )
        total_action_dim = sum(
            env.action_spaces[a].shape[0] for a in env.agents
        )
        
        agent = MADDPGAgent(
            agent_id=agent_idx,
            state_dim=state_dim,
            action_dim=action_dim,
            total_state_dim=total_state_dim,
            total_action_dim=total_action_dim,
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            gamma=config['gamma'],
            tau=config['tau'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            device=device
        )
        agents[agent_name] = agent
    
    logger.info(f"✅ Created {len(agents)} agents")
    
    # Dashboard connector - با ارسال صحیح agent_names
    save_dir = project_root / 'results' / f'level{level}' / datetime.now().strftime('%Y%m%d_%H%M%S')
    agent_names = list(agents.keys())
    dashboard = EnhancedDashboardConnector(save_dir, logger, agent_names=agent_names)
    
    # Early stopping monitor
    early_stop = EarlyStoppingMonitor(
        level=level,
        window_size=config['monitor'].window_size,
        upper_bound=config['monitor'].upper_bound,
        lower_bound=config['monitor'].lower_bound,
        patience=config['monitor'].patience,
        min_episodes=config['monitor'].min_episodes,
        check_interval=config['monitor'].check_interval
    )
    
    # System monitor
    system_monitor = SystemMonitor(save_dir / 'system_logs', logger)
    
    # Training metrics
    episode_rewards = deque(maxlen=100)
    success_rates = deque(maxlen=100)
    best_reward = float('-inf')
    epsilon = config['epsilon_start']
    total_steps = 0
    
    # Training loop
    pbar = tqdm(range(config['episodes']), desc=f"Training Level{level}")
    
    for episode in pbar:
        # Reset environment (Parallel API)
        observations, infos = env.reset()
        episode_reward = 0
        episode_steps = 0
        agent_metrics = {name: {'rewards': [], 'losses': []} for name in agents.keys()}
        
        # Episode loop
        done = False
        while not done:
            # Collect actions from all agents
            actions = {}
            for agent_name in env.agents:
                obs = observations[agent_name]
                agent = agents[agent_name]
                
                # Exploration vs Exploitation
                if episode < config['warmup_episodes'] or np.random.random() < epsilon:
                    action = env.action_spaces[agent_name].sample()
                else:
                    action = agent.select_action(obs)
                
                actions[agent_name] = action
            
            # Environment step (Parallel API)
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if episode is done
            done = all(terminations.values()) or all(truncations.values())
            
            # Store transitions
            for agent_name in env.agents:
                # Prepare global states and actions
                global_state = np.concatenate([observations[a] for a in env.agents])
                global_next_state = np.concatenate([next_observations[a] for a in env.agents])
                global_actions = np.concatenate([actions[a] for a in env.agents])
                
                # Store in replay buffer
                agents[agent_name].store_transition(
                    state=observations[agent_name],
                    action=actions[agent_name],
                    reward=rewards[agent_name],
                    next_state=next_observations[agent_name],
                    done=done,
                    global_state=global_state,
                    global_next_state=global_next_state,
                    global_actions=global_actions
                )
                
                # Track metrics
                agent_metrics[agent_name]['rewards'].append(rewards[agent_name])
            
            # Update observations
            observations = next_observations
            episode_reward += sum(rewards.values())
            episode_steps += 1
            total_steps += 1
            
            # Train agents
            if episode >= config['warmup_episodes'] and episode_steps % config['update_every'] == 0:
                for agent_name, agent in agents.items():
                    if len(agent.replay_buffer) >= agent.batch_size:
                        critic_loss, actor_loss = agent.update(list(agents.values()))
                        agent_metrics[agent_name]['losses'].append({
                            'critic': critic_loss,
                            'actor': actor_loss
                        })
        
        # Episode statistics
        episode_rewards.append(episode_reward)
        success = infos.get('escape_success', False)
        success_rates.append(1.0 if success else 0.0)
        
        # Epsilon decay
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        
        # Calculate metrics
        avg_reward = np.mean(episode_rewards)
        avg_success = np.mean(success_rates)
        
        # Agent-specific metrics
        agent_dashboard_metrics = {}
        for agent_name in agents.keys():
            losses = agent_metrics[agent_name]['losses']
            if losses:
                agent_dashboard_metrics[agent_name] = {
                    'reward': np.mean(agent_metrics[agent_name]['rewards']),
                    'actor_loss': np.mean([l['actor'] for l in losses]),
                    'critic_loss': np.mean([l['critic'] for l in losses]),
                    'actions_mean': 0.5,  # Placeholder
                    'actions_std': 0.1    # Placeholder
                }
        
        # Dashboard updates
        metrics = {
            'reward': episode_reward,
            'critic_loss': np.mean([m['critic_loss'] for m in agent_dashboard_metrics.values()]) if agent_dashboard_metrics else 0,
            'actor_loss': np.mean([m['actor_loss'] for m in agent_dashboard_metrics.values()]) if agent_dashboard_metrics else 0,
            'escape_success': success,
            'saturation_rate': 0.0,  # Calculate if needed
            'epsilon': epsilon,
            'avg_reward': avg_reward,
            'avg_success': avg_success,
            'best_reward': best_reward,
            'total_steps': total_steps,
            'total_episodes': config['episodes'],
            'drift_critic': 0.0  # Calculate if needed
        }
        
        dashboard.update_live(episode, metrics)
        dashboard.update_overview(episode, metrics)
        dashboard.update_agents(episode, agent_dashboard_metrics)
        
        # Early stopping check
        if episode > config['monitor'].min_episodes:
            avg_critic_loss = metrics['critic_loss']
            report = early_stop.check_health(episode)
            should_stop = report.get('should_stop', False)
            stop_reason = report.get('stop_reason', '')
            
            if should_stop:
                logger.warning(f"⚠️ Early stopping triggered: {stop_reason}")
                break
        
        # Update progress bar
        pbar.set_postfix({
            'reward': f"{episode_reward:.2f}",
            'avg_reward': f"{avg_reward:.2f}",
            'success': f"{avg_success*100:.1f}%",
            'epsilon': f"{epsilon:.3f}"
        })
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            checkpoint_dir = save_dir / 'checkpoints'
            checkpoint_dir.mkdir(exist_ok=True)
            
            for agent_name, agent in agents.items():
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'episode': episode,
                    'reward': best_reward
                }, checkpoint_dir / f'{agent_name}_best.pth')
        
        # Periodic saves
        if (episode + 1) % 500 == 0:
            dashboard.export_csv(episode)
    
    # Final export
    dashboard.export_csv(config['episodes'])
    env.close()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Level {level} Training Complete!")
    logger.info(f"Best Reward: {best_reward:.2f}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    # Train all levels
    for level in [1, 2, 3]:
        train_level(level)
