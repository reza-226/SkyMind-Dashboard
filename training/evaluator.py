import numpy as np
import torch

class Evaluator:
    """Evaluate agent performance"""
    
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
    
    def evaluate(self, num_episodes=10):
        """Run evaluation episodes"""
        rewards = []
        success_rates = []
        latencies = []
        energies = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_success = 0
            episode_latency = []
            episode_energy = []
            done = False
            
            while not done:
                with torch.no_grad():
                    action = self.agent.select_action(state, explore=False)
                
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if info.get('success', False):
                    episode_success += 1
                    episode_latency.append(info.get('latency', 0))
                    episode_energy.append(info.get('energy', 0))
                
                state = next_state
            
            rewards.append(episode_reward)
            success_rates.append(episode_success)
            if episode_latency:
                latencies.append(np.mean(episode_latency))
            if episode_energy:
                energies.append(np.mean(episode_energy))
        
        return {
            'avg_reward': np.mean(rewards),
            'avg_success_rate': np.mean(success_rates),
            'avg_latency': np.mean(latencies) if latencies else 0,
            'avg_energy': np.mean(energies) if energies else 0
        }
