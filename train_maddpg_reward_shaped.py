"""
train_maddpg_reward_shaped.py
Training با Reward Shaping بهبود یافته + اصلاح Action Saturation
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import logging
import random

from pettingzoo.mpe import simple_tag_v3
from configs.curriculum_config import CURRICULUM_STAGES

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RewardShaper:
    """Reward Shaping برای بهبود Training"""
    
    def __init__(self, config):
        self.distance_weight = config.get('distance_weight', 0.1)
        self.collision_penalty = config.get('collision_penalty', -5.0)
        self.escape_bonus = config.get('escape_bonus', 5.0)
        self.time_penalty = config.get('time_penalty', -0.01)
        
        self.prev_distances = {}
    
    def shape_reward(self, agent_id, obs, base_reward, done):
        """محاسبه Shaped Reward"""
        
        shaped_reward = base_reward
        
        # 1. Distance Reward (نزدیک/دور شدن)
        if agent_id.startswith('agent'):
            # Agent باید از Adversary دور شه
            adversary_pos = obs[-4:-2]  # موقعیت نسبی Adversary
            distance = np.linalg.norm(adversary_pos)
            
            if agent_id in self.prev_distances:
                prev_dist = self.prev_distances[agent_id]
                distance_change = distance - prev_dist
                
                # پاداش برای دور شدن
                shaped_reward += distance_change * self.distance_weight
            
            self.prev_distances[agent_id] = distance
            
            # Escape Bonus
            if not done and distance > 0.5:
                shaped_reward += self.escape_bonus
        
        else:  # Adversary
            # Adversary باید به Agent نزدیک شه
            agent_pos = obs[-4:-2]  # موقعیت نسبی Agent
            distance = np.linalg.norm(agent_pos)
            
            if agent_id in self.prev_distances:
                prev_dist = self.prev_distances[agent_id]
                distance_change = distance - prev_dist
                
                # پاداش برای نزدیک شدن
                shaped_reward -= distance_change * self.distance_weight
            
            self.prev_distances[agent_id] = distance
        
        # 2. Time Penalty (برای تشویق به سرعت)
        shaped_reward += self.time_penalty
        
        # 3. Collision Penalty (اگر برخورد کرد)
        if done and base_reward < 0:
            shaped_reward += self.collision_penalty
        
        return shaped_reward
    
    def reset(self):
        """ریست برای episode جدید"""
        self.prev_distances.clear()


class Actor(nn.Module):
    """
    ✅ Actor Network - اصلاح شده برای جلوگیری از Action Saturation
    تغییرات:
    - استفاده از Tanh به جای Sigmoid
    - مقیاس‌بندی به محدوده [0.025, 0.975]
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs):
        x = self.net(obs)
        
        # ✅ تغییر از Sigmoid به Tanh
        x = torch.tanh(x)  # محدوده: [-1, 1]
        
        # ✅ مقیاس‌بندی به [0.025, 0.975]
        # [-1, 1] → [0, 1] → [0.025, 0.975]
        action = (x + 1.0) / 2.0  # [-1,1] → [0,1]
        action = action * 0.95 + 0.025  # [0,1] → [0.025,0.975]
        
        return action


class Critic(nn.Module):
    """Critic Network"""
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)


class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity=1_000_000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """نمونه‌برداری تصادفی"""
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    
    def __len__(self):
        return len(self.buffer)


class MADDPGAgent:
    """
    ✅ MADDPG Agent - اصلاح شده
    تغییرات:
    - اضافه شدن Gradient Clipping با max_norm=0.5
    """
    
    def __init__(self, obs_dim, action_dim, agent_id, device, config):
        self.agent_id = agent_id
        self.device = device
        self.config = config
        
        # ✅ Gradient Clipping Threshold
        self.max_grad_norm = 0.5
        
        # Networks
        self.actor = Actor(obs_dim, action_dim, config['hidden_dim']).to(device)
        self.actor_target = Actor(obs_dim, action_dim, config['hidden_dim']).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config['lr_actor']
        )
        
        logger.info(f"[AGENT] {agent_id} initialized: "
                   f"obs_dim={obs_dim}, action_dim={action_dim}")
    
    def select_action(self, obs, explore=True, epsilon=0.1):
        """انتخاب action با exploration"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        
        # ✅ Exploration noise
        if explore:
            noise = np.random.normal(0, epsilon, size=action.shape)
            action = np.clip(action + noise, 0.025, 0.975)
        
        return action
    
    def update_target(self, tau):
        """Soft update target networks"""
        for param, target_param in zip(
            self.actor.parameters(),
            self.actor_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )


class MADDPGTrainer:
    """
    ✅ MADDPG Trainer - اصلاح شده
    تغییرات:
    - اضافه شدن Gradient Clipping در train_step
    """
    
    def __init__(self, agents, env, config):
        self.agents = {aid: agents[aid] for aid in agents}
        self.env = env
        self.config = config
        self.device = config['device']
        
        # Critic (Centralized)
        obs_dims = {aid: agents[aid].actor.net[0].in_features 
                   for aid in agents}
        action_dim = 5  # فرض: همه عامل‌ها action_dim یکسان دارند
        
        total_obs_dim = sum(obs_dims.values())
        total_action_dim = len(agents) * action_dim
        
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
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config['lr_critic']
        )
        
        # Replay Buffer
        self.buffer = ReplayBuffer(config['buffer_size'])
        
        # Reward Shaper
        self.reward_shaper = RewardShaper(config['reward_shaping'])
        
        logger.info(f"[TRAINER] Initialized with {len(agents)} agents")
        logger.info(f"[CRITIC] total_obs_dim={total_obs_dim}, "
                   f"total_action_dim={total_action_dim}")
    
    def train_step(self):
        """
        ✅ یک گام Training - اصلاح شده
        تغییرات:
        - اضافه شدن Gradient Clipping برای Actor و Critic
        """
        
        if len(self.buffer) < self.config['batch_size']:
            return None
        
        # Sample batch
        batch = self.buffer.sample(self.config['batch_size'])
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch
        
        # Convert to tensors
        obs = {aid: torch.FloatTensor(np.array([o[aid] for o in obs_batch])).to(self.device)
               for aid in self.agents}
        
        actions = {aid: torch.FloatTensor(np.array([a[aid] for a in actions_batch])).to(self.device)
                  for aid in self.agents}
        
        rewards = {aid: torch.FloatTensor(np.array([r[aid] for r in rewards_batch])).to(self.device)
                  for aid in self.agents}
        
        next_obs = {aid: torch.FloatTensor(np.array([no[aid] for no in next_obs_batch])).to(self.device)
                   for aid in self.agents}
        
        dones = torch.FloatTensor(np.array(dones_batch)).to(self.device)
        
        # Concatenate all obs and actions
        all_obs = torch.cat([obs[aid] for aid in sorted(self.agents.keys())], dim=1)
        all_actions = torch.cat([actions[aid] for aid in sorted(self.agents.keys())], dim=1)
        all_next_obs = torch.cat([next_obs[aid] for aid in sorted(self.agents.keys())], dim=1)
        
        # Update Critic
        with torch.no_grad():
            next_actions = torch.cat([
                self.agents[aid].actor_target(next_obs[aid])
                for aid in sorted(self.agents.keys())
            ], dim=1)
            
            target_q = self.critic_target(all_next_obs, next_actions)
            
            # Q-target برای همه عامل‌ها
            y = {}
            for aid in self.agents:
                y[aid] = rewards[aid].unsqueeze(1) + \
                        (1 - dones.unsqueeze(1)) * self.config['gamma'] * target_q
        
        current_q = self.critic(all_obs, all_actions)
        
        # Critic loss (میانگین برای همه عامل‌ها)
        critic_loss = sum([
            nn.MSELoss()(current_q, y[aid])
            for aid in self.agents
        ]) / len(self.agents)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # ✅ Gradient Clipping برای Critic
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        
        self.critic_optimizer.step()
        
        # Update Actors
        actor_losses = {}
        for aid in self.agents:
            # محاسبه action جدید
            new_actions = actions.copy()
            new_actions[aid] = self.agents[aid].actor(obs[aid])
            
            all_new_actions = torch.cat([
                new_actions[a] for a in sorted(self.agents.keys())
            ], dim=1)
            
            # Actor loss
            actor_loss = -self.critic(all_obs, all_new_actions).mean()
            
            self.agents[aid].actor_optimizer.zero_grad()
            actor_loss.backward()
            
            # ✅ Gradient Clipping برای Actor
            torch.nn.utils.clip_grad_norm_(
                self.agents[aid].actor.parameters(), 
                self.agents[aid].max_grad_norm
            )
            
            self.agents[aid].actor_optimizer.step()
            
            actor_losses[aid] = actor_loss.item()
        
        # Soft update target networks
        for agent in self.agents.values():
            agent.update_target(self.config['tau'])
        
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.config['tau'] * param.data +
                (1 - self.config['tau']) * target_param.data
            )
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_losses': actor_losses
        }


def train_stage(stage_config, checkpoint_dir=None):
    """
    ✅ Training یک stage - اصلاح شده
    تغییرات:
    - epsilon_start: 0.3 (بالاتر)
    - epsilon_min: 0.05 (بالاتر)
    - epsilon_decay: 0.9995 (آهسته‌تر)
    """
    
    logger.info("="*80)
    logger.info(f"🎯 Stage: {stage_config['name']}")
    logger.info("="*80)
    
    # ایجاد محیط
    env = simple_tag_v3.parallel_env(
        **stage_config['env_config'],
        max_cycles=50,
        continuous_actions=True,
        render_mode=None
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[DEVICE] Using: {device}")
    
    # دریافت ابعاد
    obs, _ = env.reset()
    agents_list = list(obs.keys())
    obs_dims = {aid: env.observation_space(aid).shape[0] for aid in agents_list}
    action_dim = env.action_space(agents_list[0]).shape[0]
    
    logger.info(f"\n📐 ابعاد محیط:")
    for aid, obs_dim in obs_dims.items():
        logger.info(f"  {aid}: obs_dim={obs_dim}")
    logger.info(f"  action_dim={action_dim}")
    
    # تنظیمات Training
    config = {
        'hidden_dim': 256,
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'batch_size': 256,
        'buffer_size': 1_000_000,
        'device': device,
        'reward_shaping': {
            'distance_weight': 1.0,
            'collision_penalty': -10.0,
            'escape_bonus': 2.0,
            'time_penalty': -0.01
        }
    }
    
    # ایجاد Agents
    agents = {}
    for aid in agents_list:
        agents[aid] = MADDPGAgent(
            obs_dims[aid],
            action_dim,
            aid,
            device,
            config
        )
    
    # بارگذاری checkpoint (اگر وجود داشت)
    if checkpoint_dir and Path(checkpoint_dir).exists():
        logger.info(f"\n📥 بارگذاری از: {checkpoint_dir}")
        for aid in agents:
            model_path = Path(checkpoint_dir) / f"{aid}.pth"
            if model_path.exists():
                agents[aid].actor.load_state_dict(
                    torch.load(model_path, map_location=device)
                )
                logger.info(f"  ✅ {aid} loaded")
    
    # Trainer
    trainer = MADDPGTrainer(agents, env, config)
    
    # Training Loop
    num_episodes = stage_config['episodes']
    logger.info(f"\n🎮 شروع Training ({num_episodes} episodes)...")
    
    # ✅ اصلاح شده: پارامترهای Epsilon بهبود یافته
    epsilon = 0.3          # شروع بالاتر (قبلی: 0.2)
    epsilon_decay = 0.9995 # کاهش آهسته‌تر (قبلی: 0.995)
    epsilon_min = 0.05     # حداقل بالاتر (قبلی: 0.03)
    
    best_mean_reward = -float('inf')
    
    for episode in tqdm(range(num_episodes), desc=f"Training {stage_config['name']}"):
        obs, _ = env.reset()
        trainer.reward_shaper.reset()
        
        episode_rewards = {aid: 0 for aid in agents}
        done = False
        step = 0
        
        while not done and step < 50:
            # انتخاب actions
            actions = {}
            for aid in agents:
                action = agents[aid].select_action(
                    obs[aid],
                    explore=True,
                    epsilon=epsilon
                )
                actions[aid] = action
            
            # گام بعدی
            next_obs, base_rewards, terminations, truncations, infos = env.step(actions)
            
            # Reward Shaping
            shaped_rewards = {}
            for aid in agents:
                shaped_rewards[aid] = trainer.reward_shaper.shape_reward(
                    aid,
                    obs[aid],
                    base_rewards[aid],
                    terminations[aid] or truncations[aid]
                )
                episode_rewards[aid] += shaped_rewards[aid]
            
            # ذخیره در buffer
            done_flag = all(terminations.values()) or all(truncations.values())
            trainer.buffer.push(obs, actions, shaped_rewards, next_obs, done_flag)
            
            # Training step
            if len(trainer.buffer) >= config['batch_size']:
                trainer.train_step()
            
            obs = next_obs
            done = done_flag
            step += 1
        
        # کاهش epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
        # Logging
        mean_reward = np.mean(list(episode_rewards.values()))
        
        if (episode + 1) % 100 == 0:
            logger.info(f"\nEpisode {episode + 1}/{num_episodes}:")
            logger.info(f"  Mean Reward: {mean_reward:.2f}")
            logger.info(f"  Epsilon: {epsilon:.4f}")
            
            for aid, rew in episode_rewards.items():
                logger.info(f"  {aid}: {rew:.2f}")
        
        # ذخیره بهترین مدل
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            save_dir = Path(f"models/{stage_config['name']}/best")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for aid in agents:
                torch.save(
                    agents[aid].actor.state_dict(),
                    save_dir / f"{aid}.pth"
                )
        
        # ذخیره checkpoint
        if (episode + 1) % 1000 == 0:
            save_dir = Path(f"models/{stage_config['name']}/checkpoint_{episode + 1}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for aid in agents:
                torch.save(
                    agents[aid].actor.state_dict(),
                    save_dir / f"{aid}.pth"
                )
            
            torch.save(
                trainer.critic.state_dict(),
                save_dir / "critic.pth"
            )
    
    # ذخیره نهایی
    save_dir = Path(f"models/{stage_config['name']}/checkpoint_final")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for aid in agents:
        torch.save(
            agents[aid].actor.state_dict(),
            save_dir / f"{aid}.pth"
        )
    
    torch.save(
        trainer.critic.state_dict(),
        save_dir / "critic.pth"
    )
    
    env.close()
    logger.info(f"\n✅ {stage_config['name']} Training Complete!")


def main():
    """اجرای کامل Curriculum"""
    
    logger.info("="*80)
    logger.info("🚀 MADDPG Training با Reward Shaping و اصلاح Action Saturation")
    logger.info("="*80)
    
    for stage in CURRICULUM_STAGES:
        train_stage(stage)


if __name__ == "__main__":
    main()
