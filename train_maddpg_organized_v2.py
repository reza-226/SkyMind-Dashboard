"""
train_maddpg_organized_v2.py
آموزش MADDPG با OutputManager، Resume و توقف بین Levelها
✅ اصلاح شده: actor_target و critic_target
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path    
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
from tqdm import tqdm

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modules
from pettingzoo_env_maddpg import MADDPGEnv
from maddpg_agent import MADDPGAgent
from early_stopping_monitor import EarlyStoppingMonitor
from system_monitor import ThresholdConfig, SystemMonitor
from utils.output_manager import OutputManager

# Configure logging
def setup_logging(log_file: Path):
    """راه‌اندازی logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


class EnhancedDashboardConnector:
    """نسخه پیشرفته Dashboard Connector"""
    
    def __init__(self, save_dir: Path, logger=None, agent_names=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        self.json_dir = self.save_dir / 'json_logs'
        self.csv_dir = self.save_dir / 'csv_exports'
        self.json_dir.mkdir(exist_ok=True)
        self.csv_dir.mkdir(exist_ok=True)
        
        if agent_names is None:
            agent_names = ['agent_0', 'agent_1', 'agent_2']
        
        self.agent_data = {agent_name: [] for agent_name in agent_names}
        self.episode_data = []
        self.prev_weights = None
        
        if self.logger:
            self.logger.info(f"✅ Dashboard initialized at {save_dir}")
    
    def calculate_action_saturation(self, actions: np.ndarray) -> float:
        threshold = 0.05
        saturated = np.sum((actions < threshold) | (actions > (1 - threshold)))
        total = actions.size
        return (saturated / total) * 100 if total > 0 else 0
    
    def calculate_weight_drift(self, agent: MADDPGAgent) -> Dict[str, float]:
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
        total_norm = 0.0
        for param in network.parameters():
            total_norm += torch.norm(param.data).item() ** 2
        return np.sqrt(total_norm)
    
    def update_live(self, episode: int, metrics: Dict):
        live_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'reward': float(metrics.get('reward', 0)),
            'critic_loss': float(metrics.get('critic_loss', 0)),
            'actor_loss': float(metrics.get('actor_loss', 0)),
            'success_rate': float(metrics.get('escape_success', 0)) * 100,
            'epsilon': float(metrics.get('epsilon', 0))
        }
        
        with open(self.json_dir / 'live_metrics.json', 'w') as f:
            json.dump(live_data, f, indent=2)
        
        self.episode_data.append(live_data)
    
    def update_overview(self, episode: int, metrics: Dict):
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
                'actor_loss': float(metrics.get('actor_loss', 0))
            }
        }
        
        with open(self.json_dir / 'overview.json', 'w') as f:
            json.dump(overview, f, indent=2)
    
    def update_agents(self, episode: int, agent_metrics: Dict):
        for agent_id, metrics in agent_metrics.items():
            agent_record = {
                'episode': episode,
                'reward': float(metrics.get('reward', 0)),
                'actor_loss': float(metrics.get('actor_loss', 0)),
                'critic_loss': float(metrics.get('critic_loss', 0))
            }
            self.agent_data[agent_id].append(agent_record)
        
        with open(self.json_dir / 'agents_data.json', 'w') as f:
            json.dump(self.agent_data, f, indent=2)
    
    def export_csv(self, episode: int):
        if not self.episode_data:
            return
        
        df = pd.DataFrame(self.episode_data)
        csv_path = self.csv_dir / f'training_ep{episode}.csv'
        df.to_csv(csv_path, index=False)
    
    def get_training_history_df(self) -> pd.DataFrame:
        """تبدیل داده‌ها به DataFrame"""
        return pd.DataFrame(self.episode_data)


def create_level_config(level: int) -> Dict:
    """ایجاد تنظیمات برای هر Level"""
    
    configs = {
        1: {
            'episodes': 1000,
            'difficulty': 'easy',
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
            'threshold_reward': -18.0,
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
            'episodes': 1500,
            'difficulty': 'medium',
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
            'threshold_reward': -15.0,
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
            'episodes': 1500,
            'difficulty': 'hard',
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
            'threshold_reward': -12.0,
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


def train_level(level: int, resume: bool = False, base_dir: str = "results"):
    """
    آموزش برای یک Level مشخص
    
    Args:
        level: شماره Level (1, 2, 3)
        resume: ادامه از checkpoint قبلی؟
        base_dir: پوشه اصلی نتایج
    """
    
    print(f"\n{'='*80}")
    print(f"🚀 LEVEL {level} TRAINING")
    print(f"{'='*80}\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load config
    config = create_level_config(level)
    
    # ========================================
    # 1. Initialize OutputManager
    # ========================================
    
    output_mgr = OutputManager(
        base_dir=base_dir,
        level=level,
        difficulty=config['difficulty'],
        resume=resume
    )
    
    # Setup logging
    logger = setup_logging(output_mgr.get_log_file())
    logger.info(f"{'='*80}")
    logger.info(f"Starting Level {level}: {config['difficulty']}")
    logger.info(f"{'='*80}")
    
    # Save config
    output_mgr.save_config(config)
    
    # ========================================
    # 2. Create Environment & Agents
    # ========================================
    
    env = MADDPGEnv(
        n_agents=config['n_agents'],
        n_adversaries=config['n_adversaries'],
        max_cycles=config['max_cycles']
    )
    
    agents = {}
    for agent_idx, agent_name in enumerate(env.agents):
        state_dim = env.observation_spaces[agent_name].shape[0]
        action_dim = env.action_spaces[agent_name].shape[0]
        
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
    
    # ========================================
    # 3. Load Checkpoint (if resume)
    # ========================================
    
    start_episode = 1
    best_reward = float('-inf')
    episode_rewards_list = []
    success_rates_list = []
    epsilon = config['epsilon_start']
    total_steps = 0
    
    if resume:
        checkpoint = output_mgr.load_checkpoint()
        if checkpoint is not None:
            start_episode = checkpoint.get('episode', 1) + 1
            best_reward = checkpoint.get('best_reward', float('-inf'))
            epsilon = checkpoint.get('epsilon', config['epsilon_start'])
            total_steps = checkpoint.get('total_steps', 0)
            
            # ✅ اصلاح: بارگذاری با نام‌های صحیح
            agent_states = checkpoint.get('agents', {})
            for agent_name, agent in agents.items():
                if agent_name in agent_states:
                    agent.actor.load_state_dict(agent_states[agent_name]['actor'])
                    agent.critic.load_state_dict(agent_states[agent_name]['critic'])
                    agent.actor_target.load_state_dict(agent_states[agent_name]['actor_target'])
                    agent.critic_target.load_state_dict(agent_states[agent_name]['critic_target'])
            
            logger.info(f"✅ Resumed from episode {start_episode-1}")
            logger.info(f"   Best reward: {best_reward:.2f}")
        else:
            logger.warning("⚠️  No checkpoint found. Starting fresh.")
    
    # ========================================
    # 4. Dashboard & Monitors
    # ========================================
    
    dashboard = EnhancedDashboardConnector(
        output_mgr.run_dir / 'dashboard',
        logger,
        agent_names=list(agents.keys())
    )
    
    early_stop = EarlyStoppingMonitor(
        level=level,
        window_size=config['monitor'].window_size,
        upper_bound=config['monitor'].upper_bound,
        lower_bound=config['monitor'].lower_bound,
        patience=config['monitor'].patience,
        min_episodes=config['monitor'].min_episodes,
        check_interval=config['monitor'].check_interval
    )
    
    system_monitor = SystemMonitor(output_mgr.run_dir / 'system_logs', logger)
    
    # ========================================
    # 5. Training Loop
    # ========================================
    
    episode_rewards = deque(maxlen=100)
    success_rates = deque(maxlen=100)
    
    pbar = tqdm(range(start_episode, config['episodes'] + 1), desc=f"Level {level}")
    
    for episode in pbar:
        
        # Reset environment
        observations, infos = env.reset()
        episode_reward = 0
        episode_steps = 0
        agent_metrics = {name: {'rewards': [], 'losses': []} for name in agents.keys()}
        
        # Episode loop
        done = False
        while not done:
            # Collect actions
            actions = {}
            for agent_name in env.agents:
                obs = observations[agent_name]
                agent = agents[agent_name]
                
                if episode < config['warmup_episodes'] or np.random.random() < epsilon:
                    action = env.action_spaces[agent_name].sample()
                else:
                    action = agent.select_action(obs)
                
                actions[agent_name] = action
            
            # Environment step
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            done = all(terminations.values()) or all(truncations.values())
            
            # Store transitions
            for agent_name in env.agents:
                global_state = np.concatenate([observations[a] for a in env.agents])
                global_next_state = np.concatenate([next_observations[a] for a in env.agents])
                global_actions = np.concatenate([actions[a] for a in env.agents])
                
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
                
                agent_metrics[agent_name]['rewards'].append(rewards[agent_name])
            
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
        
        episode_rewards_list.append(episode_reward)
        success_rates_list.append(1.0 if success else 0.0)
        
        # Epsilon decay
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        
        # Calculate metrics
        avg_reward = np.mean(episode_rewards)
        avg_success = np.mean(success_rates)
        
        # Agent metrics for dashboard
        agent_dashboard_metrics = {}
        for agent_name in agents.keys():
            losses = agent_metrics[agent_name]['losses']
            if losses:
                agent_dashboard_metrics[agent_name] = {
                    'reward': np.mean(agent_metrics[agent_name]['rewards']),
                    'actor_loss': np.mean([l['actor'] for l in losses]),
                    'critic_loss': np.mean([l['critic'] for l in losses])
                }
        
        # Dashboard updates
        metrics = {
            'reward': episode_reward,
            'critic_loss': np.mean([m['critic_loss'] for m in agent_dashboard_metrics.values()]) if agent_dashboard_metrics else 0,
            'actor_loss': np.mean([m['actor_loss'] for m in agent_dashboard_metrics.values()]) if agent_dashboard_metrics else 0,
            'escape_success': success,
            'epsilon': epsilon,
            'avg_reward': avg_reward,
            'avg_success': avg_success,
            'best_reward': best_reward,
            'total_steps': total_steps,
            'total_episodes': config['episodes']
        }
        
        dashboard.update_live(episode, metrics)
        dashboard.update_overview(episode, metrics)
        dashboard.update_agents(episode, agent_dashboard_metrics)
        
        # ========================================
        # Save Best Model
        # ========================================
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            
            # ✅ اصلاح: ذخیره با نام‌های صحیح
            model_state = {
                'episode': episode,
                'reward': best_reward,
                'level': level,
                'agents': {
                    agent_name: {
                        'actor': agent.actor.state_dict(),
                        'critic': agent.critic.state_dict(),
                        'actor_target': agent.actor_target.state_dict(),
                        'critic_target': agent.critic_target.state_dict()
                    }
                    for agent_name, agent in agents.items()
                }
            }
            
            output_mgr.save_best_model(model_state, level_best=False)
            logger.info(f"🏆 New best: {best_reward:.2f} (ep {episode})")
        
        # ========================================
        # Auto-save Checkpoint
        # ========================================
        
        if episode % 100 == 0:
            # ✅ اصلاح: ذخیره با نام‌های صحیح
            checkpoint_state = {
                'episode': episode,
                'level': level,
                'best_reward': best_reward,
                'epsilon': epsilon,
                'total_steps': total_steps,
                'agents': {
                    agent_name: {
                        'actor': agent.actor.state_dict(),
                        'critic': agent.critic.state_dict(),
                        'actor_target': agent.actor_target.state_dict(),
                        'critic_target': agent.critic_target.state_dict()
                    }
                    for agent_name, agent in agents.items()
                }
            }
            
            output_mgr.save_checkpoint(checkpoint_state, episode=episode, keep_last_n=5)
        
        # Early stopping check
        if episode > config['monitor'].min_episodes:
            report = early_stop.check_health(critic_loss=metrics['critic_loss'])
            if report.get('should_stop', False):
                logger.warning(f"⚠️ Early stop: {report.get('stop_reason')}")
                break
        
        # Update progress bar
        pbar.set_postfix({
            'R': f"{episode_reward:.1f}",
            'Avg': f"{avg_reward:.1f}",
            'Best': f"{best_reward:.1f}",
            'ε': f"{epsilon:.3f}"
        })
    
    # ========================================
    # 6. Save Final Results
    # ========================================
    
    logger.info(f"✅ Level {level} completed!")
    
    # ✅ اصلاح: ذخیره checkpoint نهایی
    final_checkpoint = {
        'episode': config['episodes'],
        'level': level,
        'best_reward': best_reward,
        'epsilon': epsilon,
        'total_steps': total_steps,
        'agents': {
            agent_name: {
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_target': agent.actor_target.state_dict(),
                'critic_target': agent.critic_target.state_dict()
            }
            for agent_name, agent in agents.items()
        }
    }
    output_mgr.save_checkpoint(final_checkpoint, is_final=True)
    
    # Save history
    history_df = dashboard.get_training_history_df()
    output_mgr.save_training_history(history_df)
    
    # Save summary
    summary = {
        'level': level,
        'difficulty': config['difficulty'],
        'total_episodes': config['episodes'],
        'best_episode': int(history_df.loc[history_df['reward'].idxmax(), 'episode']),
        'best_reward': float(best_reward),
        'final_reward': float(episode_rewards_list[-1]),
        'avg_reward': float(np.mean(episode_rewards_list)),
        'recent_100_avg': float(np.mean(episode_rewards_list[-100:])),
        'success_rate': float(np.mean(success_rates_list)) * 100,
        'threshold_reward': config['threshold_reward']
    }
    output_mgr.save_summary(summary)
    
    # Export CSV
    dashboard.export_csv(config['episodes'])
    
    # Print summary
    output_mgr.print_summary()
    
    env.close()
    
    print(f"\n{'='*80}")
    print(f"✅ LEVEL {level} COMPLETE!")
    print(f"   Best Reward: {best_reward:.2f}")
    print(f"   Threshold: {config['threshold_reward']}")
    print(f"   Success: {'YES ✓' if best_reward >= config['threshold_reward'] else 'NO ✗'}")
    print(f"{'='*80}\n")
    
    return best_reward >= config['threshold_reward']


# ========================================
# Main: Training with User Confirmation
# ========================================

def main():
    """اجرای اصلی با توقف بین Levelها"""
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_level", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--base_dir", type=str, default="results")
    parser.add_argument("--auto_continue", action="store_true", 
                        help="ادامه خودکار بدون تأیید")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 MADDPG CURRICULUM TRAINING")
    print("="*80 + "\n")
    
    for level in range(args.start_level, 4):
        
        # Train level
        success = train_level(level, resume=args.resume, base_dir=args.base_dir)
        
        # Check threshold
        if not success:
            print(f"\n⚠️  Level {level} did not reach threshold!")
            response = input("Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Training stopped by user.")
                break
        
        # Ask for continuation (unless last level or auto_continue)
        if level < 3:
            if not args.auto_continue:
                print(f"\n✅ Level {level} finished successfully!")
                response = input(f"Continue to Level {level+1}? (y/n): ").strip().lower()
                if response != 'y':
                    print(f"Training stopped after Level {level}.")
                    break
            else:
                print(f"\n✅ Auto-continuing to Level {level+1}...\n")
                time.sleep(2)
    
    print("\n" + "="*80)
    print("🎉 ALL TRAINING COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
