"""
evaluate_trained_models.py
ارزیابی مدل‌های آموزش‌دیده
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from pettingzoo.mpe import simple_tag_v3
from configs.curriculum_config import CURRICULUM_STAGES

# Import Actor از فایل training
sys.path.append(str(Path(__file__).parent))

class Actor(torch.nn.Module):
    """Actor Network"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, obs):
        return self.net(obs)


def load_models(checkpoint_dir, obs_dims, action_dim, device):
    """بارگذاری مدل‌ها از checkpoint"""
    models = {}
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint یافت نشد: {checkpoint_dir}")
        return None
    
    for model_file in checkpoint_path.glob("*.pth"):
        if model_file.stem == 'critic':
            continue
        
        agent_id = model_file.stem
        obs_dim = obs_dims[agent_id]
        
        model = Actor(obs_dim, action_dim, hidden_dim=256).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        
        models[agent_id] = model
        print(f"  ✅ {agent_id}: obs_dim={obs_dim}, action_dim={action_dim}")
    
    return models


def evaluate_stage(stage_config, checkpoint_dir, num_episodes=20):
    """ارزیابی یک stage"""
    
    print(f"\n{'='*80}")
    print(f"🎯 Stage: {stage_config['name']}")
    print(f"{'='*80}")
    
    # ایجاد محیط
    env = simple_tag_v3.parallel_env(
        num_good=stage_config['env_config']['num_good'],
        num_adversaries=stage_config['env_config']['num_adversaries'],
        num_obstacles=stage_config['env_config']['num_obstacles'],
        max_cycles=50,
        continuous_actions=True,
        render_mode=None
    )
    
    device = torch.device('cpu')
    
    # دریافت ابعاد
    obs, _ = env.reset()
    agents = list(obs.keys())
    obs_dims = {aid: env.observation_space(aid).shape[0] for aid in agents}
    action_dim = env.action_space(agents[0]).shape[0]
    
    print(f"\n📐 ابعاد محیط:")
    for aid, obs_dim in obs_dims.items():
        print(f"  {aid}: obs_dim={obs_dim}")
    print(f"  action_dim={action_dim}")
    
    # بارگذاری مدل‌ها
    print(f"\n📥 بارگذاری از: {checkpoint_dir}")
    models = load_models(checkpoint_dir, obs_dims, action_dim, device)
    
    if models is None:
        return None
    
    # ارزیابی
    print(f"\n🎮 شروع ارزیابی ({num_episodes} episode)...")
    
    episode_rewards = {aid: [] for aid in agents}
    success_count = 0  # تعداد دفعات فرار موفق
    catch_count = 0    # تعداد دفعات گرفتن
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        done = False
        step = 0
        max_steps = 50
        episode_reward = {aid: 0 for aid in agents}
        
        while not done and step < max_steps:
            # انتخاب action
            actions = {}
            for agent_id in agents:
                obs_tensor = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = models[agent_id](obs_tensor).cpu().numpy()[0]
                actions[agent_id] = action
            
            # گام بعدی
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # محاسبه reward
            for aid in agents:
                episode_reward[aid] += rewards[aid]
            
            done = all(terminations.values()) or all(truncations.values())
            step += 1
        
        # ذخیره نتایج
        for aid in agents:
            episode_rewards[aid].append(episode_reward[aid])
        
        # شمارش موفقیت/شکست
        # فرض: اگر adversary reward منفی باشه = فرار موفق
        if 'adversary_0' in agents:
            if episode_reward['adversary_0'] < 0:
                success_count += 1
            else:
                catch_count += 1
    
    env.close()
    
    # نمایش نتایج
    print(f"\n📊 نتایج ارزیابی:")
    print(f"{'='*80}")
    
    for aid in agents:
        mean_reward = np.mean(episode_rewards[aid])
        std_reward = np.std(episode_rewards[aid])
        print(f"  {aid}:")
        print(f"    Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"    Min/Max: {min(episode_rewards[aid]):.2f} / "
              f"{max(episode_rewards[aid]):.2f}")
    
    success_rate = (success_count / num_episodes) * 100
    print(f"\n  🎯 نرخ موفقیت (Escape): {success_rate:.1f}%")
    print(f"  🏃 Successful Escapes: {success_count}/{num_episodes}")
    print(f"  🎣 Catches: {catch_count}/{num_episodes}")
    print(f"{'='*80}")
    
    return {
        'stage': stage_config['name'],
        'rewards': episode_rewards,
        'success_rate': success_rate
    }


def main():
    """ارزیابی همه stages"""
    
    print("="*80)
    print("🎯 ارزیابی مدل‌های آموزش‌دیده")
    print("="*80)
    
    results = []
    
    for stage in CURRICULUM_STAGES:
        checkpoint_dir = f"models/{stage['name']}/checkpoint_final"
        
        result = evaluate_stage(stage, checkpoint_dir, num_episodes=20)
        
        if result:
            results.append(result)
    
    # خلاصه نهایی
    print(f"\n\n{'='*80}")
    print("📈 خلاصه نتایج:")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n{result['stage']}:")
        print(f"  Success Rate: {result['success_rate']:.1f}%")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
