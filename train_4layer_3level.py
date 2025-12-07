# train_4layer_3level.py (نسخه نهایی - سازگار با Actor ساده)
"""
آموزش MADDPG در 4 لایه محاسباتی × 3 سطح پیچیدگی
با action format ساده (array 7 عنصری)
✅ State dimension به صورت خودکار از محیط تشخیص داده می‌شود
✅ سازگار با Actor ساده (بدون offload_head/continuous_head)
"""
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from environments.uav_mec_env import UAVMECEnvironment
from models.actor_critic.maddpg_agent import MADDPGAgent

# =====================================================================
# تنظیمات سه سطح
# =====================================================================
TRAINING_LEVELS = {
    'level_1': {
        'name': 'ساده (Testing)',
        'config': {
            'num_uavs': 3,
            'num_devices': 5,
            'num_edge_servers': 2,
            'grid_size': 500.0,
            'max_steps': 50,
        },
        'training': {
            'episodes': 500,
            'batch_size': 64,
            'warmup': 100,
        }
    },
    'level_2': {
        'name': 'متوسط (Training)',
        'config': {
            'num_uavs': 5,
            'num_devices': 10,
            'num_edge_servers': 3,
            'grid_size': 1000.0,
            'max_steps': 100,
        },
        'training': {
            'episodes': 1000,
            'batch_size': 128,
            'warmup': 200,
        }
    },
    'level_3': {
        'name': 'پیچیده (Evaluation)',
        'config': {
            'num_uavs': 8,
            'num_devices': 15,
            'num_edge_servers': 4,
            'grid_size': 2000.0,
            'max_steps': 200,
        },
        'training': {
            'episodes': 2000,
            'batch_size': 256,
            'warmup': 500,
        }
    }
}

# =====================================================================
# Trainer Class
# =====================================================================
class MultiLevelTrainer:
    """آموزش در چند سطح"""
    
    def __init__(self, level_name='level_1'):
        self.level_name = level_name
        self.level_info = TRAINING_LEVELS[level_name]
        
        print(f"\n🎯 Training Level: {self.level_info['name']}")
        print("=" * 80)
        
        # محیط
        env_config = self.level_info['config']
        self.env = UAVMECEnvironment(**env_config)
        
        # ✅ تشخیص خودکار ابعاد واقعی از محیط
        print(f"\n🔍 Detecting actual state dimensions from environment...")
        dummy_state = self.env.reset()
        if isinstance(dummy_state, tuple):
            dummy_state = dummy_state[0]
        
        self.state_dim = len(dummy_state) if isinstance(dummy_state, np.ndarray) else dummy_state.shape[0]
        self.action_dim = 7
        
        print(f"\n{'='*60}")
        print(f"📐 Detected Dimensions:")
        print(f"   State Dimension:  {self.state_dim}")
        print(f"   Action Dimension: {self.action_dim}")
        print(f"{'='*60}")
        
        # چند بار reset کنیم برای اطمینان
        print(f"\n🧪 Verifying state dimensions (5 samples)...")
        for i in range(5):
            test_state = self.env.reset()
            if isinstance(test_state, tuple):
                test_state = test_state[0]
            dim = len(test_state)
            print(f"   Sample {i+1}: dimension = {dim}")
            if dim != self.state_dim:
                raise ValueError(f"❌ Inconsistent state dimension! Expected {self.state_dim}, got {dim}")
        print(f"✅ State dimension verified: {self.state_dim}")
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n🖥️  Device: {self.device}")
        
        # ✅ Agent با ابعاد واقعی
        print(f"\n🔧 Creating MADDPGAgent with state_dim={self.state_dim}...")
        self.agent = MADDPGAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=512,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.99,
            tau=0.01
        )
        
        # چک ابعاد Actor (safe check)
        print(f"\n{'='*60}")
        print(f"🔍 Actor Network Architecture:")
        
        # چک لایه‌های اصلی که حتماً باید وجود داشته باشند
        if hasattr(self.agent.actor, 'fc1'):
            print(f"   fc1: {self.agent.actor.fc1.in_features} → {self.agent.actor.fc1.out_features}")
        
        if hasattr(self.agent.actor, 'fc2'):
            print(f"   fc2: {self.agent.actor.fc2.in_features} → {self.agent.actor.fc2.out_features}")
        
        # چک لایه‌های اضافی (اگر وجود دارند)
        if hasattr(self.agent.actor, 'offload_head'):
            print(f"   offload_head: {self.agent.actor.offload_head.in_features} → {self.agent.actor.offload_head.out_features}")
        
        if hasattr(self.agent.actor, 'continuous_head'):
            print(f"   continuous_head: {self.agent.actor.continuous_head.in_features} → {self.agent.actor.continuous_head.out_features}")
        
        if hasattr(self.agent.actor, 'action_out'):
            print(f"   action_out: {self.agent.actor.action_out.in_features} → {self.agent.actor.action_out.out_features}")
        
        print(f"{'='*60}\n")
        
        # Verify input dimension
        if hasattr(self.agent.actor, 'fc1') and self.agent.actor.fc1.in_features != self.state_dim:
            raise ValueError(
                f"❌ Actor dimension MISMATCH!\n"
                f"   Expected state_dim: {self.state_dim}\n"
                f"   Actor fc1.in_features: {self.agent.actor.fc1.in_features}\n"
                f"   → Solution: Delete __pycache__ folders and restart!"
            )
        
        print("✅ Actor network dimensions match environment!")
        
        # ذخیره نتایج
        self.results_dir = Path(f'results/4layer_3level/{level_name}')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # ذخیره config
        config_info = {
            'level': level_name,
            'level_name': self.level_info['name'],
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'env_config': env_config,
            'training_config': self.level_info['training'],
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.results_dir / 'config.json', 'w') as f:
            json.dump(config_info, f, indent=2)
        
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': []
        }
    
    def train(self):
        """آموزش اصلی"""
        train_cfg = self.level_info['training']
        num_episodes = train_cfg['episodes']
        batch_size = train_cfg['batch_size']
        warmup = train_cfg['warmup']
        
        print(f"\n{'='*80}")
        print(f"🚀 Starting Training")
        print(f"   Episodes: {num_episodes}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Warmup Episodes: {warmup}")
        print(f"{'='*80}\n")
        
        best_reward = -float('inf')
        
        with tqdm(range(num_episodes), desc=f"📊 {self.level_name}") as pbar:
            for episode in pbar:
                # Reset environment
                state = self.env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                
                # چک ابعاد state
                if len(state) != self.state_dim:
                    raise ValueError(f"State dimension mismatch! Expected {self.state_dim}, got {len(state)}")
                
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    # انتخاب action
                    if episode < warmup:
                        action = np.random.uniform(-1, 1, size=self.action_dim)
                    else:
                        action = self.agent.select_action(state, noise=0.1)
                    
                    # اجرا در محیط
                    result = self.env.step(action)
                    
                    if len(result) == 4:
                        next_state, reward, done, info = result
                    else:
                        raise ValueError(f"Unexpected step result: {len(result)}")
                    
                    if isinstance(next_state, tuple):
                        next_state = next_state[0]
                    
                    # چک ابعاد next_state
                    if len(next_state) != self.state_dim:
                        raise ValueError(f"Next state dimension mismatch! Expected {self.state_dim}, got {len(next_state)}")
                    
                    # ذخیره در buffer
                    self.agent.replay_buffer.push(state, action, reward, next_state, done)
                    
                    # آموزش
                    if len(self.agent.replay_buffer) >= batch_size and episode >= warmup:
                        actor_loss, critic_loss = self.agent.update(batch_size=batch_size)
                        if actor_loss is not None:
                            self.history['actor_losses'].append(actor_loss)
                            self.history['critic_losses'].append(critic_loss)
                    
                    episode_reward += reward
                    episode_length += 1
                    state = next_state
                
                # ذخیره نتایج
                self.history['episode_rewards'].append(episode_reward)
                self.history['episode_lengths'].append(episode_length)
                
                # بهترین مدل
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    self.agent.save(self.results_dir / 'best_model.pth')
                
                # Progress bar
                avg_reward = np.mean(self.history['episode_rewards'][-100:])
                pbar.set_postfix({
                    'reward': f'{episode_reward:.2f}',
                    'avg': f'{avg_reward:.2f}',
                    'best': f'{best_reward:.2f}',
                    'buffer': len(self.agent.replay_buffer)
                })
                
                # Checkpoint
                if (episode + 1) % 100 == 0:
                    self._save_checkpoint(episode + 1)
        
        print(f"\n{'='*80}")
        print(f"✅ Training Complete!")
        print(f"   Best Reward: {best_reward:.2f}")
        print(f"   Final Avg (last 100): {np.mean(self.history['episode_rewards'][-100:]):.2f}")
        print(f"{'='*80}\n")
        
        self._save_final_results()
    
    def _save_checkpoint(self, episode):
        """ذخیره checkpoint"""
        ckpt_dir = self.results_dir / 'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f'checkpoint_ep{episode}.pth'
        self.agent.save(ckpt_path)
        
        with open(self.results_dir / f'history_ep{episode}.json', 'w') as f:
            json.dump({
                'episode': episode,
                'avg_reward': float(np.mean(self.history['episode_rewards'][-100:])),
                'buffer_size': len(self.agent.replay_buffer)
            }, f, indent=2)
    
    def _save_final_results(self):
        """ذخیره نتایج نهایی"""
        results = {
            'level': self.level_name,
            'config': self.level_info['config'],
            'dimensions': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            },
            'final_stats': {
                'total_episodes': len(self.history['episode_rewards']),
                'avg_reward': float(np.mean(self.history['episode_rewards'])),
                'best_reward': float(np.max(self.history['episode_rewards'])),
                'final_100_avg': float(np.mean(self.history['episode_rewards'][-100:]))
            },
            'history': {
                'episode_rewards': [float(r) for r in self.history['episode_rewards']],
                'episode_lengths': [int(l) for l in self.history['episode_lengths']]
            }
        }
        
        with open(self.results_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self._plot_results()
        print(f"💾 Results saved to: {self.results_dir}")
    
    def _plot_results(self):
        """رسم نمودارها"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward Curve
        ax = axes[0, 0]
        rewards = self.history['episode_rewards']
        window = min(50, len(rewards) // 10)
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(rewards, alpha=0.3, label='Raw', color='lightblue')
            ax.plot(smoothed, label=f'MA({window})', color='blue', linewidth=2)
        else:
            ax.plot(rewards, color='blue')
        ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Episode Length
        ax = axes[0, 1]
        ax.plot(self.history['episode_lengths'], color='green', alpha=0.7)
        ax.set_title('Episode Length', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.grid(True, alpha=0.3)
        
        # Actor Loss
        ax = axes[1, 0]
        if self.history['actor_losses']:
            ax.plot(self.history['actor_losses'], color='red', alpha=0.5)
            ax.set_title('Actor Loss', fontsize=14, fontweight='bold')
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
        
        # Critic Loss
        ax = axes[1, 1]
        if self.history['critic_losses']:
            ax.plot(self.history['critic_losses'], color='purple', alpha=0.5)
            ax.set_title('Critic Loss', fontsize=14, fontweight='bold')
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

# =====================================================================
# Main
# =====================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train MADDPG on 4-Layer UAV-MEC')
    parser.add_argument('--level', type=str, default='level_1',
                       choices=['level_1', 'level_2', 'level_3'],
                       help='Training difficulty level')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚁 UAV-MEC 4-Layer MADDPG Training")
    print("   ✅ Auto-detect state_dim from environment")
    print("="*80)
    
    trainer = MultiLevelTrainer(level_name=args.level)
    trainer.train()

if __name__ == '__main__':
    main()
