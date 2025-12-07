"""
ردیابی دقیق Episode به Episode
"""
import torch
import numpy as np
from pathlib import Path
import json
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from environments.uav_mec_env import UAVMECEnvironment
from models.actor_critic.maddpg_agent import MADDPGAgent

def adapt_state(state, target_dim):
    """تطبیق ابعاد state"""
    current_dim = state.shape[0] if len(state.shape) == 1 else state.shape[-1]
    
    if current_dim == target_dim:
        return state
    elif current_dim < target_dim:
        padding = target_dim - current_dim
        return np.concatenate([state, np.zeros(padding)])
    else:
        return state[:target_dim]

def trace_episode(model_path, episode_num=1, verbose=True):
    """ردیابی دقیق یک episode"""
    
    # بارگذاری config
    model_dir = Path(model_path).parent
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # استخراج dimensions
    if 'dimensions' in config:
        state_dim = config['dimensions']['state']
        action_dim = config['dimensions']['action']
    else:
        state_dim = config.get('state_dim', 537)
        action_dim = config.get('action_dim', 11)
    
    # ایجاد environment
    env = UAVMECEnvironment(**config['env_config'])
    
    # ایجاد agent
    agent = MADDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=512,
        lr_actor=1e-4,
        lr_critic=1e-3
    )
    
    # بارگذاری مدل
    checkpoint = torch.load(model_path, map_location='cpu')
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor.eval()
    
    # اجرای episode
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = adapt_state(state, state_dim)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 Episode {episode_num} - Detailed Trace")
        print(f"{'='*80}\n")
    
    done = False
    step = 0
    total_reward = 0
    rewards_list = []
    actions_list = []
    
    while not done and step < 50:
        with torch.no_grad():
            action = agent.select_action(state, noise=0.0)
        
        result = env.step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
        elif len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            break
        
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        next_state = adapt_state(next_state, state_dim)
        
        rewards_list.append(reward)
        actions_list.append(action)
        total_reward += reward
        
        if verbose:
            print(f"Step {step:2d} │ R: {reward:8.2f} │ Cumulative: {total_reward:10.2f}")
        
        state = next_state
        step += 1
    
    # خلاصه episode
    rewards_arr = np.array(rewards_list)
    actions_arr = np.array(actions_list)
    
    if verbose:
        print(f"\n{'-'*80}")
        print(f"📊 Episode Summary")
        print(f"{'-'*80}")
        print(f"Total Steps:    {step}")
        print(f"Total Reward:   {total_reward:.2f}")
        print(f"Mean Reward:    {rewards_arr.mean():.2f} ± {rewards_arr.std():.2f}")
        print(f"Min/Max:        {rewards_arr.min():.2f} / {rewards_arr.max():.2f}")
        print(f"{'='*80}\n")
    
    return {
        'episode': episode_num,
        'total_reward': total_reward,
        'steps': step,
        'rewards': rewards_list,
        'actions': actions_list
    }

def run_multiple_episodes(model_path, num_episodes=5):
    """اجرای چند episode و نمایش خلاصه"""
    
    print(f"\n{'#'*80}")
    print(f"# Running {num_episodes} Test Episodes")
    print(f"{'#'*80}\n")
    
    results = []
    for i in range(num_episodes):
        result = trace_episode(model_path, episode_num=i+1, verbose=(i==0))
        results.append(result['total_reward'])
        
        if i > 0:
            print(f"Episode {i+1}: Total Reward = {result['total_reward']:.2f}")
    
    # خلاصه نهایی
    results_arr = np.array(results)
    
    print(f"\n{'='*80}")
    print(f"📈 {num_episodes}-Episode Test Summary")
    print(f"{'='*80}")
    print(f"Mean:    {results_arr.mean():.2f} ± {results_arr.std():.2f}")
    print(f"Median:  {np.median(results_arr):.2f}")
    print(f"Min:     {results_arr.min():.2f}")
    print(f"Max:     {results_arr.max():.2f}")
    print(f"Range:   {results_arr.max() - results_arr.min():.2f}")
    
    # نمایش وضعیت
    positive_count = np.sum(results_arr > 0)
    print(f"\n✨ Success Rate: {positive_count}/{num_episodes} ({100*positive_count/num_episodes:.1f}%)")
    
    if results_arr.std() / abs(results_arr.mean()) > 0.5:
        print("⚠️  WARNING: High variance detected - unstable policy!")
    
    print(f"{'='*80}\n")
    
    return results

if __name__ == "__main__":
    model_path = 'results/4layer_3level/level_1/best_model.pth'
    run_multiple_episodes(model_path, num_episodes=10)
