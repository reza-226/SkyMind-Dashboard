# train_improved.py
"""
Improved Training Script with:
- Early Stopping
- Best Model Tracking
- Periodic Evaluation
- Curriculum Learning (optional)
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from improved_config import TRAINING_CONFIG, CURRICULUM_CONFIG

class ImprovedTrainer:
    def __init__(self, env, agent, config, use_curriculum=False):
        self.env = env
        self.agent = agent
        self.config = config
        self.use_curriculum = use_curriculum
        
        # Tracking
        self.best_reward = -float('inf')
        self.best_episode = 0
        self.no_improvement_count = 0
        self.training_history = []
        
        # Paths
        self.save_dir = Path('results/improved_training')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self):
        print("🚀 Starting Improved Training...")
        print(f"Config: {self.config}")
        
        if self.use_curriculum:
            return self._train_curriculum()
        else:
            return self._train_standard()
    
    def _train_standard(self):
        """Standard training with early stopping"""
        total_episodes = self.config['episodes']
        
        for episode in range(total_episodes):
            # Training episode
            episode_reward = self._run_episode(episode)
            
            # Update history
            self.training_history.append({
                'episode': episode,
                'reward': episode_reward,
                'noise_scale': self.agent.noise_scale
            })
            
            # Check for improvement
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.best_episode = episode
                self.no_improvement_count = 0
                self._save_best_model(episode, episode_reward)
                print(f"✅ New best! Episode {episode}: {episode_reward:.2f}")
            else:
                self.no_improvement_count += 1
            
            # Early stopping check
            if self.no_improvement_count >= self.config['patience']:
                print(f"⚠️ Early stopping at episode {episode}")
                print(f"No improvement for {self.config['patience']} episodes")
                break
            
            # Periodic evaluation
            if episode % self.config['eval_interval'] == 0:
                self._evaluate(episode)
            
            # Periodic save
            if episode % self.config['save_interval'] == 0:
                self._save_checkpoint(episode)
            
            # Decay noise
            self.agent.decay_noise()
        
        # Final save
        self._save_final_results()
        return self.training_history
    
    def _train_curriculum(self):
        """Training with curriculum learning"""
        print("📚 Using Curriculum Learning")
        stages = CURRICULUM_CONFIG['stages']
        
        for stage_idx, stage in enumerate(stages):
            print(f"\n{'='*50}")
            print(f"Stage {stage_idx+1}: {stage['name']}")
            print(f"Episodes: {stage['episodes']}, Difficulty: {stage['difficulty']}")
            print(f"{'='*50}\n")
            
            # Adjust environment difficulty if needed
            # self.env.set_difficulty(stage['difficulty'])
            
            # Adjust noise
            self.agent.noise_scale = stage['noise_scale']
            
            # Train for this stage
            for episode in range(stage['episodes']):
                global_episode = sum(s['episodes'] for s in stages[:stage_idx]) + episode
                
                episode_reward = self._run_episode(global_episode)
                
                self.training_history.append({
                    'episode': global_episode,
                    'reward': episode_reward,
                    'stage': stage['name'],
                    'noise_scale': self.agent.noise_scale
                })
                
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.best_episode = global_episode
                    self._save_best_model(global_episode, episode_reward)
                
                if episode % 10 == 0:
                    recent_avg = np.mean([h['reward'] for h in self.training_history[-10:]])
                    print(f"Stage {stage['name']} - Episode {episode}/{stage['episodes']}: "
                          f"Reward={episode_reward:.2f}, Avg10={recent_avg:.2f}")
        
        self._save_final_results()
        return self.training_history
    
    def _run_episode(self, episode):
        """Run single training episode"""
        state = self.env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Get action from agent
            action = self.agent.select_action(state, explore=True)
            
            # Environment step
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            if len(self.agent.replay_buffer) > self.config['batch_size']:
                self.agent.update()
            
            episode_reward += reward
            state = next_state
            step += 1
        
        return episode_reward
    
    def _evaluate(self, episode):
        """Evaluate current policy (no exploration)"""
        eval_rewards = []
        for _ in range(5):  # 5 evaluation episodes
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, explore=False)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            
            eval_rewards.append(episode_reward)
        
        avg_eval = np.mean(eval_rewards)
        print(f"📊 Evaluation at episode {episode}: {avg_eval:.2f} ± {np.std(eval_rewards):.2f}")
        
    def _save_best_model(self, episode, reward):
        """Save best model so far"""
        path = self.save_dir / 'best_model.pth'
        torch.save({
            'episode': episode,
            'reward': reward,
            'actor_state': self.agent.actor.state_dict(),
            'critic_state': self.agent.critic.state_dict(),
            'actor_target_state': self.agent.actor_target.state_dict(),
            'critic_target_state': self.agent.critic_target.state_dict(),
        }, path)
    
    def _save_checkpoint(self, episode):
        """Save periodic checkpoint"""
        path = self.save_dir / f'checkpoint_ep{episode}.pth'
        torch.save({
            'episode': episode,
            'actor_state': self.agent.actor.state_dict(),
            'critic_state': self.agent.critic.state_dict(),
            'training_history': self.training_history,
        }, path)
    
    def _save_final_results(self):
        """Save final training results"""
        results = {
            'config': self.config,
            'best_episode': self.best_episode,
            'best_reward': float(self.best_reward),
            'total_episodes': len(self.training_history),
            'history': self.training_history
        }
        
        with open(self.save_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Training completed!")
        print(f"Best episode: {self.best_episode}")
        print(f"Best reward: {self.best_reward:.2f}")
        print(f"Results saved to: {self.save_dir}")


# Main training script
if __name__ == '__main__':
    from your_environment import YourEnvironment  # جایگزین کن
    from your_agent import MADDPGAgent  # جایگزین کن
    
    # Initialize
    env = YourEnvironment()
    agent = MADDPGAgent(
        state_dim=537,
        action_dim=11,
        lr_actor=TRAINING_CONFIG['lr_actor'],
        lr_critic=TRAINING_CONFIG['lr_critic'],
        tau=TRAINING_CONFIG['tau'],
        gamma=TRAINING_CONFIG['gamma'],
        buffer_size=TRAINING_CONFIG['buffer_size']
    )
    
    # Train
    trainer = ImprovedTrainer(
        env=env,
        agent=agent,
        config=TRAINING_CONFIG,
        use_curriculum=CURRICULUM_CONFIG['use_curriculum']
    )
    
    history = trainer.train()
