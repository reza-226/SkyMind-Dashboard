"""
MADDPG Training Script with Comprehensive Metrics
اسکریپت آموزش MADDPG با متریک‌های جامع
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
from collections import deque
from typing import Dict, List, Any, Tuple

# Import custom modules
from utils.output_manager import OutputManager
from utils.metrics_collector import EnvironmentMetricsCollector
from utils.early_stopping import EarlyStopping
from utils.replay_buffer import ReplayBuffer

# فرض: محیط و شبکه‌های عصبی از قبل تعریف شده‌اند
# from environments.uav_mec_env import UAVMECEnvironment
# from networks.maddpg_networks import MADDPGActor, MADDPGCritic


class MADDPGTrainer:
    """
    کلاس اصلی برای آموزش MADDPG با متریک‌های تفصیلی
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: دیکشنری تنظیمات شامل:
                - env_config: تنظیمات محیط
                - network_config: تنظیمات شبکه
                - training_config: تنظیمات آموزش
                - output_config: تنظیمات خروجی
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ایجاد محیط
        self.env = self._create_environment()
        self.n_agents = self.env.n_agents
        
        # ایجاد شبکه‌ها
        self.actors = []
        self.critics = []
        self.actor_targets = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        self._initialize_networks()
        
        # Replay Buffer
        buffer_size = config['training_config'].get('buffer_size', 100000)
        self.replay_buffer = ReplayBuffer(buffer_size, self.n_agents)
        
        # Output Manager
        output_cfg = config.get('output_config', {})
        self.output_manager = OutputManager(
            base_dir=output_cfg.get('base_dir', 'results'),
            level=output_cfg.get('level', 1),
            difficulty=output_cfg.get('difficulty', 'easy'),
        )
        
        # Metrics Collector
        self.metrics_collector = EnvironmentMetricsCollector()
        
        # Early Stopping
        es_cfg = config['training_config'].get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get('patience', 100),
            min_episodes=es_cfg.get('min_episodes', 200),
            metric_threshold=es_cfg.get('metric_threshold', 0.1),
        )
        
        # Training parameters
        self.gamma = config['training_config'].get('gamma', 0.99)
        self.tau = config['training_config'].get('tau', 0.01)
        self.batch_size = config['training_config'].get('batch_size', 128)
        
        # Exploration
        self.epsilon = config['training_config'].get('epsilon_start', 1.0)
        self.epsilon_decay = config['training_config'].get('epsilon_decay', 0.995)
        self.epsilon_min = config['training_config'].get('epsilon_min', 0.01)
        
        # Best model tracking
        self.best_reward = float('-inf')
        self.best_episode = 0
        
        # Save config
        self.output_manager.save_config(config)
        
        print(f"✅ MADDPGTrainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Agents: {self.n_agents}")
        print(f"   Output: {self.output_manager.run_dir}")
    
    def _create_environment(self):
        """ایجاد محیط شبیه‌سازی"""
        # این قسمت باید با محیط واقعی شما جایگزین شود
        from environments.uav_mec_env import UAVMECEnvironment
        env_cfg = self.config.get('env_config', {})
        return UAVMECEnvironment(**env_cfg)
    
    def _initialize_networks(self):
        """ایجاد و مقداردهی اولیه شبکه‌ها"""
        net_cfg = self.config.get('network_config', {})
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        for i in range(self.n_agents):
            # Actor
            actor = MADDPGActor(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=net_cfg.get('actor_hidden', [256, 128]),
            ).to(self.device)
            
            actor_target = MADDPGActor(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=net_cfg.get('actor_hidden', [256, 128]),
            ).to(self.device)
            actor_target.load_state_dict(actor.state_dict())
            
            # Critic
            critic = MADDPGCritic(
                state_dim=state_dim * self.n_agents,
                action_dim=action_dim * self.n_agents,
                hidden_dims=net_cfg.get('critic_hidden', [256, 128]),
            ).to(self.device)
            
            critic_target = MADDPGCritic(
                state_dim=state_dim * self.n_agents,
                action_dim=action_dim * self.n_agents,
                hidden_dims=net_cfg.get('critic_hidden', [256, 128]),
            ).to(self.device)
            critic_target.load_state_dict(critic.state_dict())
            
            # Optimizers
            lr = net_cfg.get('learning_rate', 1e-3)
            actor_opt = optim.Adam(actor.parameters(), lr=lr)
            critic_opt = optim.Adam(critic.parameters(), lr=lr)
            
            # Save
            self.actors.append(actor)
            self.actor_targets.append(actor_target)
            self.critics.append(critic)
            self.critic_targets.append(critic_target)
            self.actor_optimizers.append(actor_opt)
            self.critic_optimizers.append(critic_opt)
        
        print(f"✅ Networks initialized for {self.n_agents} agents")
    
    def select_actions(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        انتخاب اکشن برای تمام عامل‌ها
        
        Args:
            states: آرایه حالت‌ها [n_agents, state_dim]
            add_noise: آیا نویز exploration اضافه شود
            
        Returns:
            آرایه اکشن‌ها [n_agents, action_dim]
        """
        actions = []
        
        for i in range(self.n_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.actors[i](state).cpu().numpy()[0]
            
            # Exploration noise
            if add_noise:
                noise = np.random.normal(0, self.epsilon, size=action.shape)
                action = np.clip(action + noise, -1, 1)
            
            actions.append(action)
        
        return np.array(actions)
    
    def update_networks(self) -> Tuple[float, float]:
        """
        به‌روزرسانی شبکه‌های Actor و Critic
        
        Returns:
            (actor_loss, critic_loss)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0
        
        # نمونه‌برداری از buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # تبدیل به Tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for i in range(self.n_agents):
            # ============ Update Critic ============
            # محاسبه target Q-value
            with torch.no_grad():
                next_actions = []
                for j in range(self.n_agents):
                    next_action = self.actor_targets[j](next_states[:, j])
                    next_actions.append(next_action)
                
                next_actions = torch.cat(next_actions, dim=1)
                next_states_flat = next_states.view(self.batch_size, -1)
                
                target_q = self.critic_targets[i](next_states_flat, next_actions)
                target_q = rewards[:, i].unsqueeze(1) + \
                          self.gamma * target_q * (1 - dones[:, i].unsqueeze(1))
            
            # محاسبه current Q-value
            states_flat = states.view(self.batch_size, -1)
            actions_flat = actions.view(self.batch_size, -1)
            current_q = self.critics[i](states_flat, actions_flat)
            
            # Critic loss
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            # بک‌پراپ
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critic_optimizers[i].step()
            
            total_critic_loss += critic_loss.item()
            
            # ============ Update Actor ============
            # محاسبه اکشن‌های جدید
            new_actions = []
            for j in range(self.n_agents):
                if j == i:
                    new_action = self.actors[i](states[:, i])
                else:
                    new_action = actions[:, j]
                new_actions.append(new_action)
            
            new_actions = torch.cat(new_actions, dim=1)
            
            # Actor loss (maximize Q)
            actor_loss = -self.critics[i](states_flat, new_actions).mean()
            
            # بک‌پراپ
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()
            
            total_actor_loss += actor_loss.item()
            
            # ============ Soft Update Target Networks ============
            self._soft_update(self.actors[i], self.actor_targets[i])
            self._soft_update(self.critics[i], self.critic_targets[i])
        
        avg_actor_loss = total_actor_loss / self.n_agents
        avg_critic_loss = total_critic_loss / self.n_agents
        
        return avg_actor_loss, avg_critic_loss
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update از source به target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def train_episode(self) -> Dict[str, Any]:
        """
        اجرای یک اپیزود آموزشی
        
        Returns:
            دیکشنری شامل متریک‌های اپیزود
        """
        states = self.env.reset()
        done = False
        episode_reward = 0
        episode_rewards = [0] * self.n_agents
        step_count = 0
        
        # ریست metrics collector
        self.metrics_collector.reset()
        
        while not done:
            # انتخاب اکشن
            actions = self.select_actions(states, add_noise=True)
            
            # اجرای اکشن
            next_states, rewards, done, infos = self.env.step(actions)
            
            # جمع‌آوری متریک‌های محیط
            step_metrics = self.metrics_collector.collect_step_metrics(self.env, infos)
            
            # ذخیره در replay buffer
            self.replay_buffer.push(states, actions, rewards, next_states, done)
            
            # به‌روزرسانی شبکه‌ها
            actor_loss, critic_loss = self.update_networks()
            
            # آپدیت state
            states = next_states
            episode_reward += np.mean(rewards)
            
            for i in range(self.n_agents):
                episode_rewards[i] += rewards[i]
            
            step_count += 1
        
        # خلاصه متریک‌های اپیزود
        episode_summary = self.metrics_collector.get_episode_summary()
        
        # ترکیب با متریک‌های آموزش
        episode_metrics = {
            'episode_reward': episode_reward,
            'agent_rewards': episode_rewards,
            'avg_reward': np.mean(episode_rewards),
            'steps': step_count,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'epsilon': self.epsilon,
            **episode_summary,
        }
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return episode_metrics
    
    def train(self, n_episodes: int, save_interval: int = 100):
        """
        آموزش اصلی
        
        Args:
            n_episodes: تعداد اپیزودها
            save_interval: فاصله ذخیره checkpoint
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting MADDPG Training")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        reward_window = deque(maxlen=100)
        
        for episode in range(1, n_episodes + 1):
            episode_start = time.time()
            
            # اجرای یک اپیزود
            metrics = self.train_episode()
            
            # آپدیت window
            reward_window.append(metrics['avg_reward'])
            
            # اضافه کردن متریک‌های اضافی
            metrics['episode'] = episode
            metrics['reward_ma_100'] = np.mean(reward_window)
            metrics['reward_std_100'] = np.std(reward_window)
            metrics['elapsed_time'] = time.time() - start_time
            metrics['episode_time'] = time.time() - episode_start
            
            # ذخیره متریک‌ها
            self.output_manager.save_training_history(metrics)
            self.output_manager.save_detailed_metrics(episode, metrics)
            
            # نمایش پیشرفت
            if episode % 10 == 0:
                self._print_progress(episode, metrics, reward_window)
            
            # ذخیره checkpoint
            if episode % save_interval == 0:
                self._save_checkpoint(episode, metrics)
            
            # بررسی بهترین مدل
            if metrics['avg_reward'] > self.best_reward:
                self.best_reward = metrics['avg_reward']
                self.best_episode = episode
                self._save_best_model(episode, metrics)
            
            # بررسی early stopping
            should_stop = self.early_stopping.check_health({
                'episode': episode,
                'critic_loss': metrics['critic_loss'],
            })
            
            if should_stop:
                print(f"\n⚠️ Early stopping triggered at episode {episode}")
                self._save_checkpoint(episode, metrics)
                break
        
        # پایان آموزش
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✅ Training completed!")
        print(f"   Total episodes: {episode}")
        print(f"   Best episode: {self.best_episode}")
        print(f"   Best reward: {self.best_reward:.4f}")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"{'='*80}\n")
        
        # تولید گزارش‌ها
        self._finalize_training()
    
    def _print_progress(self, episode: int, metrics: Dict, reward_window: deque):
        """نمایش پیشرفت آموزش"""
        print(f"Episode {episode:5d} | "
              f"Reward: {metrics['avg_reward']:8.2f} | "
              f"MA-100: {np.mean(reward_window):8.2f} | "
              f"Loss(A/C): {metrics['actor_loss']:6.3f}/{metrics['critic_loss']:6.3f} | "
              f"ε: {self.epsilon:.3f} | "
              f"Time: {metrics['episode_time']:.1f}s")
        
        # نمایش متریک‌های تخصصی (اختیاری)
        if 'energy_summary' in metrics:
            energy = metrics['energy_summary']['total_energy']
            print(f"         Energy: {energy:.1f}J | ", end='')
        
        if 'task_summary' in metrics:
            success_rate = metrics['task_summary'].get('success_rate', 0)
            print(f"Task Success: {success_rate:.1f}% | ", end='')
        
        if 'safety_summary' in metrics:
            collisions = metrics['safety_summary']['total_collisions']
            print(f"Collisions: {collisions}")
        else:
            print()
    
    def _save_checkpoint(self, episode: int, metrics: Dict):
        """ذخیره checkpoint"""
        checkpoint = {
            'episode': episode,
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actor_targets': [target.state_dict() for target in self.actor_targets],
            'critic_targets': [target.state_dict() for target in self.critic_targets],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers],
            'replay_buffer': self.replay_buffer.get_state(),
            'epsilon': self.epsilon,
            'best_reward': self.best_reward,
            'metrics': metrics,
        }
        
        self.output_manager.save_checkpoint(checkpoint, episode)
    
    def _save_best_model(self, episode: int, metrics: Dict):
        """ذخیره بهترین مدل"""
        checkpoint = {
            'episode': episode,
            'actors': [actor.state_dict() for actor in self.actors],
            'actor_targets': [target.state_dict() for target in self.actor_targets],
            'metrics': metrics,
        }
        
        self.output_manager.save_best_model(
            checkpoint, 
            episode, 
            metrics['avg_reward']
        )
    
    def _finalize_training(self):
        """نهایی‌سازی و تولید گزارش‌ها"""
        print("\n📊 Generating analysis reports...")
        
        # تولید نمودارها
        self.output_manager.generate_training_plots()
        
        # تولید گزارش تحلیلی
        report = self.output_manager.generate_analysis_report()
        
        # پاکسازی checkpoint های قدیمی
        self.output_manager.cleanup_old_checkpoints(keep_last_n=5)
        
        print(f"✅ All reports saved to: {self.output_manager.run_dir}")


# ========================================
# Main Execution
# ========================================

def main():
    """تابع اصلی اجرا"""
    
    # تنظیمات
    config = {
        'env_config': {
            'n_uavs': 3,
            'n_users': 10,
            'area_size': 1000,
            'max_steps': 200,
        },
        
        'network_config': {
            'actor_hidden': [256, 128],
            'critic_hidden': [256, 128],
            'learning_rate': 1e-3,
        },
        
        'training_config': {
            'n_episodes': 4000,
            'batch_size': 128,
            'buffer_size': 100000,
            'gamma': 0.99,
            'tau': 0.01,
            'epsilon_start': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'save_interval': 100,
            
            'early_stopping': {
                'patience': 100,
                'min_episodes': 200,
                'metric_threshold': 0.1,
            },
        },
        
        'output_config': {
            'base_dir': 'results',
            'level': 1,
            'difficulty': 'easy',
        },
    }
    
    # ایجاد trainer
    trainer = MADDPGTrainer(config)
    
    # شروع آموزش
    trainer.train(
        n_episodes=config['training_config']['n_episodes'],
        save_interval=config['training_config']['save_interval'],
    )


if __name__ == "__main__":
    main()
