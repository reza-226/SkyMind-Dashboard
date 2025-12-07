# train_maddpg_final.py
"""
✅ MADDPG Training for UAVMECEnvironment
✅ با OutputManager، Dashboard، Early Stopping، Resume
✅ سازگار با signature: (num_uavs, num_devices, num_edge_servers, grid_size, max_steps)
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging
from collections import deque
from tqdm import tqdm

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ✅ استفاده از ماژول‌های موجود
from environments.uav_mec_env import UAVMECEnvironment
from models.actor_critic.maddpg_agent import MADDPGAgent
from models.action_decoder import ActionDecoder

# ✅ فرض می‌کنیم این ماژول‌ها در پروژه شما هستند
try:
    from utils.output_manager import OutputManager
    from early_stopping_monitor import EarlyStoppingMonitor
    from system_monitor import SystemMonitor, ThresholdConfig
except ImportError:
    print("⚠️  OutputManager/EarlyStop modules not found. Using basic logging.")
    OutputManager = None


# ========================================
# Logging Setup
# ========================================

def setup_logging(log_file: Path):
    """راه‌اندازی logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


# ========================================
# Configuration
# ========================================

def create_config(level: int = 1) -> Dict:
    """تنظیمات آموزش"""
    
    configs = {
        1: {
            'episodes': 1000,
            'difficulty': 'easy',
            
            # ✅ Environment params (سازگار با signature)
            'num_uavs': 8,
            'num_devices': 10,
            'num_edge_servers': 2,
            'grid_size': 1000.0,
            'max_steps': 100,
            
            # Agent params
            'buffer_size': 50000,
            'batch_size': 256,
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'gamma': 0.95,
            'tau': 0.005,
            
            # Exploration
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995,
            'noise_std': 0.1,
            
            # Training
            'warmup_episodes': 100,
            'update_every': 4,
            'target_update_every': 10,
            
            # Monitoring
            'log_every': 10,
            'save_every': 100,
            'eval_every': 50,
        },
        2: {
            'episodes': 1500,
            'difficulty': 'medium',
            'num_uavs': 8,
            'num_devices': 15,
            'num_edge_servers': 3,
            'grid_size': 1500.0,
            'max_steps': 150,
            'batch_size': 512,
            'lr_actor': 8e-5,
            'lr_critic': 8e-4,
            'warmup_episodes': 150,
            # ... (بقیه پارامترها)
        },
        3: {
            'episodes': 2000,
            'difficulty': 'hard',
            'num_uavs': 10,
            'num_devices': 20,
            'num_edge_servers': 4,
            'grid_size': 2000.0,
            'max_steps': 200,
            'batch_size': 1024,
            'lr_actor': 5e-5,
            'lr_critic': 5e-4,
            'warmup_episodes': 200,
            # ...
        }
    }
    
    return configs.get(level, configs[1])


# ========================================
# Simple Dashboard (fallback)
# ========================================

class SimpleDashboard:
    """Dashboard ساده در صورت عدم وجود OutputManager"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_data = []
    
    def update_live(self, episode: int, metrics: Dict):
        self.episode_data.append({
            'episode': episode,
            **metrics
        })
    
    def update_overview(self, episode: int, metrics: Dict):
        pass
    
    def update_agents(self, episode: int, metrics: Dict):
        pass
    
    def export_csv(self, episode: int):
        import pandas as pd
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            df.to_csv(self.save_dir / f'training_{episode}.csv', index=False)


# ========================================
# Main Training Function
# ========================================

def train_level(level: int, resume: bool = False, base_dir: str = "results"):
    """
    آموزش MADDPG برای یک Level
    
    Args:
        level: شماره Level (1, 2, 3)
        resume: ادامه از checkpoint؟
        base_dir: پوشه ذخیره نتایج
    """
    
    print(f"\n{'='*80}")
    print(f"🚀 LEVEL {level} TRAINING - UAV-MEC MADDPG")
    print(f"{'='*80}\n")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Config
    config = create_config(level)
    
    # ========================================
    # Setup Output Manager
    # ========================================
    
    if OutputManager:
        output_mgr = OutputManager(
            base_dir=base_dir,
            level=level,
            difficulty=config['difficulty'],
            resume=resume
        )
        logger = setup_logging(output_mgr.get_log_file())
        save_dir = output_mgr.run_dir
    else:
        save_dir = Path(base_dir) / f"level_{level}" / datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logging(save_dir / 'training.log')
        output_mgr = None
    
    logger.info(f"{'='*80}")
    logger.info(f"Starting Level {level}: {config['difficulty']}")
    logger.info(f"{'='*80}")
    
    # ========================================
    # Create Environment
    # ========================================
    
    env = UAVMECEnvironment(
        num_uavs=config['num_uavs'],
        num_devices=config['num_devices'],
        num_edge_servers=config['num_edge_servers'],
        grid_size=config['grid_size'],
        max_steps=config['max_steps']
    )
    
    logger.info(f"✅ Environment created:")
    logger.info(f"   UAVs: {config['num_uavs']}")
    logger.info(f"   Devices: {config['num_devices']}")
    logger.info(f"   Edge Servers: {config['num_edge_servers']}")
    logger.info(f"   Grid: {config['grid_size']}m")
    
    # ========================================
    # Create Agent
    # ========================================
    
    # ✅ فرض: state_dim=114, action_dim=11 (از تحلیل قبلی)
    state_dim = 114
    action_dim = 11  # 4 offload (one-hot) + 7 continuous
    
    agent = MADDPGAgent(
        agent_id=0,
        state_dim=state_dim,
        action_dim=action_dim,
        total_state_dim=state_dim,  # در single-agent همان state_dim
        total_action_dim=action_dim,
        lr_actor=config['lr_actor'],
        lr_critic=config['lr_critic'],
        gamma=config['gamma'],
        tau=config['tau'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        device=device
    )
    
    logger.info(f"✅ Agent created:")
    logger.info(f"   State dim: {state_dim}")
    logger.info(f"   Action dim: {action_dim}")
    
    # ========================================
    # Load Checkpoint (if resume)
    # ========================================
    
    start_episode = 1
    best_reward = float('-inf')
    epsilon = config['epsilon_start']
    total_steps = 0
    
    if resume and output_mgr:
        checkpoint = output_mgr.load_checkpoint()
        if checkpoint:
            start_episode = checkpoint.get('episode', 1) + 1
            best_reward = checkpoint.get('best_reward', float('-inf'))
            epsilon = checkpoint.get('epsilon', config['epsilon_start'])
            total_steps = checkpoint.get('total_steps', 0)
            
            agent.actor.load_state_dict(checkpoint['actor'])
            agent.critic.load_state_dict(checkpoint['critic'])
            agent.actor_target.load_state_dict(checkpoint['actor_target'])
            agent.critic_target.load_state_dict(checkpoint['critic_target'])
            
            logger.info(f"✅ Resumed from episode {start_episode-1}")
            logger.info(f"   Best reward: {best_reward:.2f}")
    
    # ========================================
    # Dashboard
    # ========================================
    
    if output_mgr:
        dashboard = SimpleDashboard(save_dir / 'dashboard')
    else:
        dashboard = SimpleDashboard(save_dir)
    
    # ========================================
    # Training Loop
    # ========================================
    
    episode_rewards = deque(maxlen=100)
    episode_losses = deque(maxlen=100)
    
    pbar = tqdm(range(start_episode, config['episodes'] + 1), desc=f"Level {level}")
    
    for episode in pbar:
        
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        critic_losses = []
        actor_losses = []
        
        done = False
        while not done:
            
            # ✅ Select action
            if episode < config['warmup_episodes'] or np.random.random() < epsilon:
                # Random exploration
                action_dict = {
                    'offload_decision': np.random.randint(0, 4),
                    'cpu_allocation': np.random.random(),
                    'bandwidth': np.random.dirichlet(np.ones(4)),
                    'movement': np.random.uniform(-1, 1, size=2)
                }
            else:
                # Policy action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    offload_logits, continuous = agent.actor(state_tensor)
                    
                    # Decode action
                    offload_idx = offload_logits.argmax(dim=-1).item()
                    cont = continuous.squeeze(0).cpu().numpy()
                    
                    action_dict = {
                        'offload_decision': offload_idx,
                        'cpu_allocation': float(cont[0]),
                        'bandwidth': cont[1:5] / cont[1:5].sum(),
                        'movement': cont[5:7]
                    }
            
            # ✅ Environment step
            next_state, reward, done, info = env.step(action_dict)
            
            # ✅ Store transition
            # تبدیل action_dict به tensor format
            action_tensor = torch.zeros(11)
            action_tensor[action_dict['offload_decision']] = 1.0  # one-hot [0:4]
            action_tensor[4] = action_dict['cpu_allocation']  # [4]
            action_tensor[5:9] = torch.tensor(action_dict['bandwidth'])  # [5:9]
            action_tensor[9:11] = torch.tensor(action_dict['movement'])  # [9:11]
            
            agent.store_transition(
                state=state,
                action=action_tensor.numpy(),
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            # ✅ Update networks
            if (episode >= config['warmup_episodes'] and 
                len(agent.replay_buffer) >= agent.batch_size and
                episode_steps % config['update_every'] == 0):
                
                critic_loss, actor_loss = agent.update([agent])  # single agent
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if done:
                break
        
        # ========================================
        # Episode Statistics
        # ========================================
        
        episode_rewards.append(episode_reward)
        
        avg_reward = np.mean(episode_rewards)
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
        
        # Epsilon decay
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        
        # ========================================
        # Logging & Dashboard
        # ========================================
        
        if episode % config['log_every'] == 0:
            logger.info(f"Episode {episode}/{config['episodes']} | "
                       f"Reward: {episode_reward:.2f} | "
                       f"Avg(100): {avg_reward:.2f} | "
                       f"ε: {epsilon:.3f}")
        
        dashboard.update_live(episode, {
            'reward': episode_reward,
            'avg_reward': avg_reward,
            'critic_loss': avg_critic_loss,
            'actor_loss': avg_actor_loss,
            'epsilon': epsilon,
            'steps': episode_steps
        })
        
        # ========================================
        # Save Best Model
        # ========================================
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            
            checkpoint = {
                'episode': episode,
                'reward': best_reward,
                'level': level,
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_target': agent.actor_target.state_dict(),
                'critic_target': agent.critic_target.state_dict(),
            }
            
            if output_mgr:
                output_mgr.save_best_model(checkpoint, level_best=False)
            else:
                torch.save(checkpoint, save_dir / 'best_model.pt')
            
            logger.info(f"🏆 New best: {best_reward:.2f}")
        
        # ========================================
        # Save Checkpoint
        # ========================================
        
        if episode % config['save_every'] == 0:
            checkpoint = {
                'episode': episode,
                'level': level,
                'best_reward': best_reward,
                'epsilon': epsilon,
                'total_steps': total_steps,
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_target': agent.actor_target.state_dict(),
                'critic_target': agent.critic_target.state_dict(),
            }
            
            if output_mgr:
                output_mgr.save_checkpoint(checkpoint, episode=episode)
            else:
                torch.save(checkpoint, save_dir / f'checkpoint_ep{episode}.pt')
        
        # Update progress bar
        pbar.set_postfix({
            'R': f"{episode_reward:.1f}",
            'Avg': f"{avg_reward:.1f}",
            'Best': f"{best_reward:.1f}",
            'ε': f"{epsilon:.3f}"
        })
    
    # ========================================
    # Final Save
    # ========================================
    
    logger.info(f"✅ Level {level} completed!")
    logger.info(f"   Best reward: {best_reward:.2f}")
    
    # Export CSV
    dashboard.export_csv(config['episodes'])
    
    print(f"\n{'='*80}")
    print(f"✅ LEVEL {level} COMPLETE!")
    print(f"   Best Reward: {best_reward:.2f}")
    print(f"{'='*80}\n")
    
    env.close()
    
    return best_reward


# ========================================
# Main Entry Point
# ========================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MADDPG Training for UAV-MEC")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--base_dir", type=str, default="results")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 MADDPG TRAINING - UAV-MEC ENVIRONMENT")
    print("="*80 + "\n")
    
    best_reward = train_level(
        level=args.level,
        resume=args.resume,
        base_dir=args.base_dir
    )
    
    print(f"\n🎉 Training Complete! Best Reward: {best_reward:.2f}\n")


if __name__ == "__main__":
    main()
