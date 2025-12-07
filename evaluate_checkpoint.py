import torch
import numpy as np
from pathlib import Path
from uav_offloading.envs.simple_uav_env import UAVOffloadingEnv

def evaluate_checkpoint(checkpoint_path, num_episodes=50):
    print(f"🎮 Evaluating: {checkpoint_path}\n")
    
    cp = Path(checkpoint_path)
    agent_0 = torch.load(cp / "agent_0.pth", map_location='cpu')
    agent_1 = torch.load(cp / "agent_1.pth", map_location='cpu')
    
    env = UAVOffloadingEnv(num_uavs=2, num_tasks=15, map_size=200, max_steps=100)
    
    class PolicyNet(torch.nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(17, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 128), torch.nn.LayerNorm(128), torch.nn.ReLU(),
                torch.nn.Linear(128, 64), torch.nn.LayerNorm(64), torch.nn.ReLU(),
                torch.nn.Linear(64, 5), torch.nn.Tanh()
            )
            self.load_state_dict(state_dict)
            self.eval()
        
        def forward(self, x):
            return self.net(x)
    
    policy_0, policy_1 = PolicyNet(agent_0), PolicyNet(agent_1)
    rewards = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0
        
        for _ in range(100):
            with torch.no_grad():
                a0 = ((policy_0(torch.FloatTensor(obs[0]).unsqueeze(0)).squeeze(0).numpy() + 1) / 2)
                a1 = ((policy_1(torch.FloatTensor(obs[1]).unsqueeze(0)).squeeze(0).numpy() + 1) / 2)
            
            obs, r, done, _ = env.step([a0, a1])
            ep_reward += sum(r)
            if done: break
        
        rewards.append(ep_reward)
        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1}: Mean = {np.mean(rewards[-10:]):.2f}")
    
    mean = np.mean(rewards)
    print(f"\n{'='*60}")
    print(f"Mean: {mean:>10.2f} | Std: {np.std(rewards):>8.2f}")
    print(f"Min:  {np.min(rewards):>10.2f} | Max: {np.max(rewards):>8.2f}")
    print(f"{'='*60}\n")
    
    status = "✅ عالی" if mean > -50 else "⚠️ متوسط" if mean > -100 else "❌ Collapse"
    print(f"وضعیت: {status}\n")
    
    return mean

if __name__ == "__main__":
    evaluate_checkpoint("models/level2/checkpoint_7000", 50)
