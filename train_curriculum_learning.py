"""
MADDPG with Curriculum Learning - Fixed Version
مرحله‌بندی آموزش برای بهبود یادگیری
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
from datetime import datetime

# تنظیم logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# تنظیم device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[DEVICE] Using device: {device}")

# Patch NumPy compatibility
logger.info("[PATCH] Applying numpy compatibility patches...")
if not hasattr(np, 'int'):
    np.int = int
    logger.info("[OK] Patched: np.int -> int")
if not hasattr(np, 'float'):
    np.float = float
    logger.info("[OK] Patched: np.float -> float")
if not hasattr(np, 'bool'):
    np.bool = bool
    logger.info("[OK] Patched: np.bool -> bool")

# =============================================================================
# Actor Network
# =============================================================================
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

# =============================================================================
# Critic Network
# =============================================================================
class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# =============================================================================
# Replay Buffer
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# =============================================================================
# MADDPG Agent
# =============================================================================
class MADDPGAgent:
    def __init__(self, obs_dim, action_dim, agent_id, n_agents, 
                 lr_actor=0.0001, lr_critic=0.001, gamma=0.95, tau=0.01):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        
        # Actor networks
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.actor_target = Actor(obs_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks
        total_obs_dim = obs_dim * n_agents
        total_action_dim = action_dim * n_agents
        self.critic = Critic(total_obs_dim, total_action_dim).to(device)
        self.critic_target = Critic(total_obs_dim, total_action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
    def select_action(self, obs, explore=True, noise_scale=0.1):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
        
        if explore:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise
            
        return np.clip(action, -1.0, 1.0)
    
    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

# =============================================================================
# MADDPG Trainer
# =============================================================================
class MADDPGTrainer:
    def __init__(self, n_agents, obs_dim, action_dim, buffer_size=100000, 
                 batch_size=64, lr_actor=0.0001, lr_critic=0.001, 
                 gamma=0.95, tau=0.01):
        self.n_agents = n_agents
        self.batch_size = batch_size
        
        # فقط پارامترهای مرتبط با Agent را پاس می‌دهیم
        agent_config = {
            'lr_actor': lr_actor,
            'lr_critic': lr_critic,
            'gamma': gamma,
            'tau': tau
        }
        
        self.agents = [
            MADDPGAgent(obs_dim, action_dim, i, n_agents, **agent_config)
            for i in range(n_agents)
        ]
        self.replay_buffer = ReplayBuffer(buffer_size)
        
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
            
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for agent_id, agent in enumerate(self.agents):
            # Prepare batch data
            obs = torch.FloatTensor([s[agent_id] for s in states]).to(device)
            act = torch.FloatTensor([a[agent_id] for a in actions]).to(device)
            rew = torch.FloatTensor([r[agent_id] for r in rewards]).unsqueeze(1).to(device)
            next_obs = torch.FloatTensor([s[agent_id] for s in next_states]).to(device)
            done = torch.FloatTensor([d[agent_id] for d in dones]).unsqueeze(1).to(device)
            
            # Global observations and actions
            all_obs = torch.FloatTensor([np.concatenate(s) for s in states]).to(device)
            all_acts = torch.FloatTensor([np.concatenate(a) for a in actions]).to(device)
            all_next_obs = torch.FloatTensor([np.concatenate(s) for s in next_states]).to(device)
            
            # Critic update
            with torch.no_grad():
                next_actions = []
                for i, ag in enumerate(self.agents):
                    next_act = ag.actor_target(torch.FloatTensor([s[i] for s in next_states]).to(device))
                    next_actions.append(next_act)
                all_next_acts = torch.cat(next_actions, dim=1)
                
                target_q = agent.critic_target(all_next_obs, all_next_acts)
                y = rew + agent.gamma * target_q * (1 - done)
            
            current_q = agent.critic(all_obs, all_acts)
            critic_loss = nn.MSELoss()(current_q, y)
            
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
            agent.critic_optimizer.step()
            
            # Actor update
            current_actions = []
            for i, ag in enumerate(self.agents):
                if i == agent_id:
                    current_actions.append(agent.actor(obs))
                else:
                    current_actions.append(torch.FloatTensor([a[i] for a in actions]).to(device))
            all_current_acts = torch.cat(current_actions, dim=1)
            
            actor_loss = -agent.critic(all_obs, all_current_acts).mean()
            
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()
            
            # Soft update target networks
            agent.soft_update(agent.actor_target, agent.actor)
            agent.soft_update(agent.critic_target, agent.critic)
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
        
        return total_actor_loss / self.n_agents, total_critic_loss / self.n_agents

# =============================================================================
# Curriculum Environment Wrapper
# =============================================================================
class CurriculumWrapper:
    """
    تنظیم سطح دشواری محیط بر اساس مرحله آموزش
    """
    def __init__(self, base_env):
        self.env = base_env
        self.difficulty = 'easy'
        
    def set_difficulty(self, level):
        """تنظیم سطح دشواری"""
        self.difficulty = level
        logger.info(f"🎚️ Difficulty set to: {level.upper()}")
        
    def reset(self):
        """Reset با تنظیمات مربوط به سطح دشواری"""
        obs, info = self.env.reset()
        
        # اعمال محدودیت‌ها بر اساس سطح
        if self.difficulty == 'easy':
            # محیط ساده‌تر: کمتر عامل، کمتر پیچیدگی
            pass
        elif self.difficulty == 'medium':
            # محیط متوسط
            pass
        elif self.difficulty == 'hard':
            # محیط کامل
            pass
            
        return obs, info
    
    def step(self, actions):
        return self.env.step(actions)
    
    @property
    def agents(self):
        return self.env.agents
    
    @property
    def possible_agents(self):
        return self.env.possible_agents

# =============================================================================
# Training با Curriculum Learning
# =============================================================================
def train_curriculum(episodes_per_stage):
    """
    آموزش تدریجی با سه مرحله
    """
    logger.info("="*80)
    logger.info("[START] MADDPG Curriculum Learning Training")
    logger.info("="*80)
    
    # Load environment
    logger.info("[ENV] Loading environment...")
    from pettingzoo.mpe import simple_spread_v3
    base_env = simple_spread_v3.parallel_env(N=2, continuous_actions=True)
    env = CurriculumWrapper(base_env)
    logger.info("[OK] Environment loaded")
    
    # Detect dimensions
    obs, _ = env.reset()
    agent_names = list(obs.keys())
    obs_dim = len(obs[agent_names[0]])
    action_dim = 5
    n_agents = len(agent_names)
    
    logger.info(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}, n_agents={n_agents}")
    
    # Initialize trainer با config کامل
    trainer = MADDPGTrainer(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr_actor=0.00005,
        lr_critic=0.0005,
        gamma=0.99,
        tau=0.005,
        buffer_size=200000,
        batch_size=128
    )
    logger.info("[OK] Trainer initialized")
    
    # Training history
    history = {}
    episode_count = 0
    best_reward = float('-inf')
    
    # Curriculum stages
    stages = [
        {'name': 'easy', 'episodes': episodes_per_stage[0], 'noise': 0.3},
        {'name': 'medium', 'episodes': episodes_per_stage[1], 'noise': 0.2},
        {'name': 'hard', 'episodes': episodes_per_stage[2], 'noise': 0.1}
    ]
    
    for stage_idx, stage_config in enumerate(stages):
        stage_name = stage_config['name']
        stage_episodes = stage_config['episodes']
        noise_scale = stage_config['noise']
        
        logger.info("="*80)
        logger.info(f"📚 STAGE {stage_idx + 1}/3: {stage_name.upper()}")
        logger.info(f"   Episodes: {stage_episodes}, Noise: {noise_scale}")
        logger.info("="*80)
        
        env.set_difficulty(stage_name)
        
        for ep in range(stage_episodes):
            obs, _ = env.reset()
            episode_reward = {agent: 0 for agent in agent_names}
            step_count = 0
            actor_loss, critic_loss = 0.0, 0.0
            
            while env.agents:
                # Select actions
                actions = {}
                for agent in env.agents:
                    agent_id = agent_names.index(agent)
                    action = trainer.agents[agent_id].select_action(
                        obs[agent], explore=True, noise_scale=noise_scale
                    )
                    actions[agent] = action
                
                # Environment step
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                
                # Store transition
                state = [obs[agent] for agent in agent_names]
                action = [actions.get(agent, np.zeros(action_dim)) for agent in agent_names]
                reward = [rewards.get(agent, 0) for agent in agent_names]
                next_state = [next_obs.get(agent, np.zeros(obs_dim)) for agent in agent_names]
                done = [terminations.get(agent, False) or truncations.get(agent, False) 
                       for agent in agent_names]
                
                trainer.replay_buffer.push(state, action, reward, next_state, done)
                
                # Update networks
                if len(trainer.replay_buffer) >= trainer.batch_size:
                    a_loss, c_loss = trainer.update()
                    if a_loss is not None:
                        actor_loss = a_loss
                        critic_loss = c_loss
                
                # Update episode reward
                for agent in agent_names:
                    if agent in rewards:
                        episode_reward[agent] += rewards[agent]
                
                obs = next_obs
                step_count += 1
                
                if step_count > 100:
                    break
            
            # Episode stats
            avg_reward = np.mean(list(episode_reward.values()))
            episode_count += 1
            
            # Save to history
            history[str(episode_count)] = {
                'episode': episode_count,
                'stage': stage_name,
                'avg_reward': float(avg_reward),
                'rewards': {k: float(v) for k, v in episode_reward.items()},
                'actor_loss': float(actor_loss),
                'critic_loss': float(critic_loss)
            }
            
            # Update best reward
            if avg_reward > best_reward:
                best_reward = avg_reward
                logger.info(f"🏆 New best reward: {best_reward:.2f}")
            
            # Log progress
            if (ep + 1) % 100 == 0:
                logger.info(
                    f"[{stage_name.upper()}] Episode {episode_count} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Best: {best_reward:.2f}"
                )
    
    # Save results
    save_dir = "models/maddpg"
    os.makedirs(save_dir, exist_ok=True)
    
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"[SAVE] History saved to {history_path}")
    
    # Save models
    for i, agent in enumerate(trainer.agents):
        torch.save(agent.actor.state_dict(), 
                  os.path.join(save_dir, f"actor_agent{i}_final.pth"))
        torch.save(agent.critic.state_dict(),
                  os.path.join(save_dir, f"critic_agent{i}_final.pth"))
    logger.info("[SAVE] Models saved")
    
    logger.info("="*80)
    logger.info(f"✅ Training complete! Best reward: {best_reward:.2f}")
    logger.info("="*80)

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    episodes_config = {
        0: 1000,  # Easy
        1: 1000,  # Medium
        2: 2000   # Hard
    }
    
    train_curriculum(episodes_config)
