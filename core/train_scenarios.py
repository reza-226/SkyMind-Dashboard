"""
Training Script for Different Obstacle Scenarios
================================================
اسکریپت آموزش برای سه سناریوی مختلف موانع
"""

import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, List
import json

from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPGAgent


class ScenarioTrainer:
    """کلاس آموزش برای سناریوهای مختلف"""
    
    def __init__(
        self,
        scenario_name: str,
        obstacle_mode: str,
        n_episodes: int = 1000,
        max_steps: int = 500,
        save_dir: str = "results/scenarios"
    ):
        self.scenario_name = scenario_name
        self.obstacle_mode = obstacle_mode
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.save_dir = Path(save_dir) / scenario_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # ایجاد محیط
        self.env = MultiUAVEnv(
            n_uavs=3,
            map_size=100.0,
            obstacle_mode=obstacle_mode,
            max_steps=max_steps,
            seed=42
        )
        
        # ایجاد عامل MADDPG
        obs_dim = list(self.env.observation_space.values())[0].shape[0]
        act_dim = list(self.env.action_space.values())[0].shape[0]
        
        self.agent = MADDPGAgent(
            n_agents=3,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=128,
            lr_actor=1e-4,
            lr_critic=1e-3,
            gamma=0.99,
            tau=0.01
        )
        
        # ذخیره متریک‌ها
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'collisions': [],
            'tasks_completed': [],
            'energy_consumed': [],
            'actor_losses': [],
            'critic_losses': [],
            'collision_risks': []
        }
        
        print(f"✅ ScenarioTrainer برای '{scenario_name}' آماده شد")
        print(f"   حالت موانع: {obstacle_mode}")
        print(f"   تعداد اپیزودها: {n_episodes}")
    
    def train(self):
        """آموزش کامل"""
        print(f"\n{'='*70}")
        print(f"🚀 شروع آموزش سناریو: {self.scenario_name}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        best_reward = -np.inf
        
        for episode in range(self.n_episodes):
            episode_reward, episode_metrics = self._run_episode(episode)
            
            # ذخیره متریک‌ها
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_lengths'].append(episode_metrics['steps'])
            self.metrics['collisions'].append(episode_metrics['collisions'])
            self.metrics['tasks_completed'].append(episode_metrics['tasks'])
            self.metrics['energy_consumed'].append(episode_metrics['energy'])
            
            if episode_metrics['actor_loss'] is not None:
                self.metrics['actor_losses'].append(episode_metrics['actor_loss'])
                self.metrics['critic_losses'].append(episode_metrics['critic_loss'])
            
            self.metrics['collision_risks'].append(episode_metrics['avg_risk'])
            
            # لاگ پیشرفت
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.metrics['episode_rewards'][-50:])
                avg_collisions = np.mean(self.metrics['collisions'][-50:])
                avg_tasks = np.mean(self.metrics['tasks_completed'][-50:])
                
                elapsed = time.time() - start_time
                eta = (elapsed / (episode + 1)) * (self.n_episodes - episode - 1)
                
                print(f"📊 Episode {episode + 1}/{self.n_episodes}")
                print(f"   ├─ Avg Reward (50): {avg_reward:.2f}")
                print(f"   ├─ Avg Collisions: {avg_collisions:.2f}")
                print(f"   ├─ Avg Tasks: {avg_tasks:.2f}")
                print(f"   ├─ Actor Loss: {episode_metrics['actor_loss']:.4f}" if episode_metrics['actor_loss'] else "")
                print(f"   ├─ Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                print(f"   └─ {'─'*50}")
                
                # ذخیره بهترین مدل
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self._save_checkpoint('best_model.pt')
                    print(f"   💾 بهترین مدل ذخیره شد! (Reward: {best_reward:.2f})")
        
        # ذخیره نهایی
        self._save_checkpoint('final_model.pt')
        self._save_metrics()
        self._generate_plots()
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ آموزش کامل شد!")
        print(f"   زمان کل: {total_time/60:.1f} دقیقه")
        print(f"   بهترین Reward: {best_reward:.2f}")
        print(f"   فایل‌ها در: {self.save_dir}")
        print(f"{'='*70}\n")
    
    def _run_episode(self, episode: int) -> tuple:
        """اجرای یک اپیزود"""
        obs, info = self.env.reset(seed=42 + episode)
        
        episode_reward = 0
        total_collisions = 0
        total_tasks = 0
        total_energy = 0
        total_risk = 0
        steps = 0
        
        actor_losses = []
        critic_losses = []
        
        for step in range(self.max_steps):
            # انتخاب عمل
            actions = {}
            for i in range(3):
                obs_i = obs[f'agent_{i}']
                action = self.agent.select_action(i, obs_i, add_noise=True)
                actions[f'agent_{i}'] = action
            
            # اجرای عمل
            next_obs, rewards, dones, infos = self.env.step(actions)
            
            # ذخیره در replay buffer
            for i in range(3):
                self.agent.store_transition(
                    obs[f'agent_{i}'],
                    actions[f'agent_{i}'],
                    rewards[f'agent_{i}'],
                    next_obs[f'agent_{i}'],
                    dones[f'agent_{i}']
                )
            
            # آموزش عامل
            if self.agent.can_update():
                losses = self.agent.update()
                if losses:
                    actor_losses.append(losses['actor_loss'])
                    critic_losses.append(losses['critic_loss'])
            
            # به‌روزرسانی متریک‌ها
            episode_reward += sum(rewards.values())
            for i in range(3):
                total_collisions += 1 if infos[f'agent_{i}']['collision'] else 0
                total_tasks += infos[f'agent_{i}']['tasks_completed']
                total_energy += infos[f'agent_{i}']['energy_consumed']
                total_risk += infos[f'agent_{i}']['collision_risk']
            
            obs = next_obs
            steps += 1
            
            if all(dones.values()):
                break
        
        return episode_reward, {
            'steps': steps,
            'collisions': total_collisions,
            'tasks': total_tasks,
            'energy': total_energy,
            'avg_risk': total_risk / (steps * 3),
            'actor_loss': np.mean(actor_losses) if actor_losses else None,
            'critic_loss': np.mean(critic_losses) if critic_losses else None
        }
    
    def _save_checkpoint(self, filename: str):
        """ذخیره checkpoint"""
        checkpoint = {
            'scenario_name': self.scenario_name,
            'obstacle_mode': self.obstacle_mode,
            'episode': len(self.metrics['episode_rewards']),
            'agent_state': self.agent.state_dict(),
            'metrics': self.metrics
        }
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def _save_metrics(self):
        """ذخیره متریک‌ها"""
        # NPZ format
        np.savez(
            self.save_dir / 'training_metrics.npz',
            episode_rewards=np.array(self.metrics['episode_rewards']),
            episode_lengths=np.array(self.metrics['episode_lengths']),
            collisions=np.array(self.metrics['collisions']),
            tasks_completed=np.array(self.metrics['tasks_completed']),
            energy_consumed=np.array(self.metrics['energy_consumed']),
            actor_losses=np.array(self.metrics['actor_losses']),
            critic_losses=np.array(self.metrics['critic_losses']),
            collision_risks=np.array(self.metrics['collision_risks'])
        )
        
        # JSON format برای خوانایی
        summary = {
            'scenario': self.scenario_name,
            'obstacle_mode': self.obstacle_mode,
            'total_episodes': self.n_episodes,
            'final_avg_reward': float(np.mean(self.metrics['episode_rewards'][-100:])),
            'best_reward': float(np.max(self.metrics['episode_rewards'])),
            'avg_collisions': float(np.mean(self.metrics['collisions'])),
            'avg_tasks_completed': float(np.mean(self.metrics['tasks_completed'])),
            'total_energy_consumed': float(np.sum(self.metrics['energy_consumed']))
        }
        
        with open(self.save_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 متریک‌ها ذخیره شدند: {self.save_dir}")
    
    def _generate_plots(self):
        """تولید نمودارها"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'نتایج آموزش - {self.scenario_name}', 
                    fontsize=16, weight='bold', y=0.995)
        
        # 1. Episode Rewards
        ax = axes[0, 0]
        rewards = self.metrics['episode_rewards']
        ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) > 50:
            window = 50
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, 
                   color='red', linewidth=2, label=f'MA({window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Collisions
        ax = axes[0, 1]
        collisions = self.metrics['collisions']
        ax.plot(collisions, alpha=0.4, color='orange')
        if len(collisions) > 50:
            smoothed = np.convolve(collisions, np.ones(50)/50, mode='valid')
            ax.plot(range(49, len(collisions)), smoothed, 
                   color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Number of Collisions')
        ax.set_title('برخوردها در طول آموزش')
        ax.grid(True, alpha=0.3)
        
        # 3. Tasks Completed
        ax = axes[0, 2]
        tasks = self.metrics['tasks_completed']
        ax.plot(tasks, alpha=0.4, color='green')
        if len(tasks) > 50:
            smoothed = np.convolve(tasks, np.ones(50)/50, mode='valid')
            ax.plot(range(49, len(tasks)), smoothed, 
                   color='darkgreen', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Tasks')
        ax.set_title('وظایف تکمیل شده')
        ax.grid(True, alpha=0.3)
        
        # 4. Energy Consumption
        ax = axes[1, 0]
        energy = self.metrics['energy_consumed']
        ax.plot(energy, alpha=0.4, color='purple')
        if len(energy) > 50:
            smoothed = np.convolve(energy, np.ones(50)/50, mode='valid')
            ax.plot(range(49, len(energy)), smoothed, 
                   color='darkviolet', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy (J)')
        ax.set_title('مصرف انرژی')
        ax.grid(True, alpha=0.3)
        
        # 5. Actor Loss
        ax = axes[1, 1]
        if self.metrics['actor_losses']:
            losses = self.metrics['actor_losses']
            ax.plot(losses, alpha=0.4, color='red')
            if len(losses) > 50:
                smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
                ax.plot(range(49, len(losses)), smoothed, 
                       color='darkred', linewidth=2)
        ax.set_xlabel('Update Step')
        ax.set_ylabel('Loss')
        ax.set_title('Actor Loss')
        ax.grid(True, alpha=0.3)
        
        # 6. Collision Risk
        ax = axes[1, 2]
        risks = self.metrics['collision_risks']
        ax.plot(risks, alpha=0.4, color='brown')
        if len(risks) > 50:
            smoothed = np.convolve(risks, np.ones(50)/50, mode='valid')
            ax.plot(range(49, len(risks)), smoothed, 
                   color='maroon', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Risk')
        ax.set_title('میانگین ریسک برخورد')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 نمودارها ذخیره شدند: {self.save_dir / 'training_curves.png'}")


def main():
    """اجرای آموزش برای هر سه سناریو"""
    
    scenarios = [
        {
            'name': 'scenario_none',
            'obstacle_mode': 'none',
            'description': 'بدون مانع (Baseline)'
        },
        {
            'name': 'scenario_moderate',
            'obstacle_mode': 'moderate',
            'description': 'موانع متوسط (3-5 ثابت)'
        },
        {
            'name': 'scenario_complex',
            'obstacle_mode': 'complex',
            'description': 'موانع پیچیده (8-10 ثابت + 2-3 متحرک)'
        }
    ]
    
    print("\n" + "="*70)
    print("🎯 آموزش سیستم SkyMind با سناریوهای مختلف موانع")
    print("="*70 + "\n")
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'🔷'*35}")
        print(f"سناریو {i}/3: {scenario['name']}")
        print(f"توضیح: {scenario['description']}")
        print(f"{'🔷'*35}\n")
        
        trainer = ScenarioTrainer(
            scenario_name=scenario['name'],
            obstacle_mode=scenario['obstacle_mode'],
            n_episodes=1000,
            max_steps=500
        )
        
        trainer.train()
        
        print(f"\n✅ سناریو {scenario['name']} با موفقیت تکمیل شد!\n")
    
    print("\n" + "="*70)
    print("🎉 تمام سناریوها آموزش داده شدند!")
    print("📁 نتایج در پوشه 'results/scenarios/' ذخیره شدند")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
