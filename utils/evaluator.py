"""
Evaluator for Multi-Agent MADDPG
"""

import torch
import numpy as np
from typing import Dict, List


class MADDPGEvaluator:
    """Evaluator for MADDPG agents"""
    
    def __init__(self, env, agents, n_eval_episodes=5, max_steps=200, device='cpu'):
        """
        Args:
            env: Multi-agent environment (MultiUAVWrapper)
            agents: Dictionary of agents {agent_id: MADDPGAgent}
            n_eval_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            device: torch device
        """
        self.env = env
        self.agents = agents
        self.agent_names = list(agents.keys())
        self.n_episodes = n_eval_episodes
        self.max_steps = max_steps
        self.device = device
        
        print(f"🔍 Evaluator initialized with agents: {self.agent_names}")
    
    def evaluate(self):
        """
        Evaluate all agents
        
        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(self.n_episodes):
            # ✅ observation is a dict: {agent_id: numpy_array}
            observation, info = self.env.reset()
            
            episode_reward = 0
            steps = 0
            done = False
            truncated = False
            
            while not (done or truncated) and steps < self.max_steps:
                actions = {}
                
                # ✅ For each agent, extract its observation from dict
                for agent_id in self.agent_names:
                    agent_obs = observation[agent_id]  # numpy array
                    
                    # Convert to tensor
                    state_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                    
                    # Select action (no noise during evaluation)
                    action = self.agents[agent_id].select_action(
                        state_tensor,
                        add_noise=False
                    )
                    
                    actions[agent_id] = action
                
                # ✅ Decode actions to environment format
                from models.actor_critic.action_decoder import ActionDecoder
                decoder = ActionDecoder()
                
                # Convert actions dict to numpy array
                action_list = [actions[agent_id] for agent_id in self.agent_names]
                action_array = np.array(action_list)
                
                # Decode to environment format
                env_actions = decoder.decode_batch(action_array)
                
                # Step environment
                observation, rewards, terminated, truncated, info = self.env.step(env_actions)
                
                # Sum rewards
                total_reward = sum([rewards[agent_id] for agent_id in self.agent_names])
                episode_reward += total_reward
                
                steps += 1
                done = terminated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        # Compute statistics
        return {
            'avg_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'avg_length': float(np.mean(episode_lengths)),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }


# ✅ Alias for backward compatibility
Evaluator = MADDPGEvaluator
