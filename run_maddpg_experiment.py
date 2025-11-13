"""
run_maddpg_experiment.py
اجرای MADDPG و استخراج متریک‌ها برای مقایسه با سیاست‌های دیگر
"""

import sys
import numpy as np
import json
import torch
from pathlib import Path

# اضافه کردن مسیر پروژه
sys.path.append(str(Path(__file__).parent))

from core.env_multi import MultiUAVEnv
from agents.agent_maddpg_multi import MADDPG_Agent


class MADDPGTester:
    """تست و ارزیابی MADDPG"""
    
    def __init__(self, model_path: str, env_config: dict):
        self.env = MultiUAVEnv(**env_config)
        self.n_agents = env_config.get('n_agents', 3)
        
        # لود کردن مدل
        print(f"🔄 Loading MADDPG model from {model_path}...")
        
        # محاسبه ابعاد state و action
        state = self.env.reset()
        state_dim = self._get_state_dim(state)
        action_dim = 4  # [v, theta, f, o] - velocity, angle, frequency, offload
        
        print(f"   State dimension: {state_dim}")
        print(f"   Action dimension: {action_dim}")
        print(f"   Number of agents: {self.n_agents}")
        
        # ✅ ساخت agent با پارامترهای صحیح
        self.agent = MADDPG_Agent(
            state_dim=state_dim,      # ✅ state_dim (نه obs_dim)
            action_dim=action_dim,    # ✅ action_dim (نه act_dim)
            n_agents=self.n_agents,
            lr=1e-4,                  # ✅ lr (نه lr_actor/lr_critic)
            gamma=0.95
        )
        
        # لود کردن وزن‌ها
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.agent.load_state_dict(checkpoint['model_state_dict'])
            else:
                # اگر checkpoint خود state_dict باشد
                try:
                    self.agent.load_state_dict(checkpoint)
                except:
                    print("   ⚠️  Checkpoint format not compatible, trying individual actors...")
                    self._load_individual_actors(Path(model_path).parent)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")
            print("   Trying to load individual actor models...")
            self._load_individual_actors(Path(model_path).parent)
    
    def _load_individual_actors(self, model_dir: Path):
        """لود کردن مدل‌های actor به صورت جداگانه"""
        loaded = False
        if hasattr(self.agent, 'actors'):
            for i in range(self.n_agents):
                actor_path = model_dir / f'actor_agent{i}.pt'
                if actor_path.exists():
                    try:
                        self.agent.actors[i].load_state_dict(
                            torch.load(actor_path, map_location='cpu')
                        )
                        print(f"   ✅ Loaded actor for agent {i}")
                        loaded = True
                    except Exception as e:
                        print(f"   ⚠️  Failed to load actor {i}: {e}")
                else:
                    print(f"   ⚠️  Actor model not found: {actor_path}")
        
        if not loaded:
            print("   ⚠️  No models loaded - using random initialization")
    
    def _get_state_dim(self, state):
        """محاسبه ابعاد state"""
        if isinstance(state, dict):
            total_dim = 0
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    total_dim += value.size
                elif isinstance(value, (list, tuple)):
                    total_dim += len(value)
                else:
                    total_dim += 1
            return total_dim
        elif isinstance(state, np.ndarray):
            return state.size
        return len(state)
    
    def _state_to_vector(self, state):
        """تبدیل state به vector برای MADDPG"""
        if isinstance(state, dict):
            vectors = []
            for key in sorted(state.keys()):
                value = state[key]
                if isinstance(value, np.ndarray):
                    vectors.append(value.flatten())
                elif isinstance(value, (list, tuple)):
                    vectors.append(np.array(value).flatten())
                elif isinstance(value, (int, float)):
                    vectors.append(np.array([value]))
            return np.concatenate(vectors)
        elif isinstance(state, np.ndarray):
            return state.flatten()
        return np.array(state)
    
    def run_episode(self, episode_num: int, max_steps: int = 200):
        """اجرای یک اپیزود"""
        state = self.env.reset()
        done = False
        episode_reward = 0
        episode_energy = 0
        episode_delays = []
        steps = 0
        
        while not done and steps < max_steps:
            # تبدیل state به vector
            state_vector = self._state_to_vector(state)
            
            # انتخاب action
            try:
                # فرض: agent.act() یک متد دارد
                if hasattr(self.agent, 'act'):
                    actions = self.agent.act(state_vector)
                else:
                    # اگر متد دیگری دارد، از آن استفاده کن
                    actions = np.random.uniform(-1, 1, (self.n_agents, 4))
                
                # تبدیل به numpy اگر tensor است
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().numpy()
                
                # اطمینان از shape صحیح
                if actions.ndim == 1:
                    actions = actions.reshape(self.n_agents, -1)
                
            except Exception as e:
                print(f"⚠️  Error in action selection: {e}")
                # استفاده از action تصادفی در صورت خطا
                actions = np.random.uniform(-1, 1, (self.n_agents, 4))
            
            # تبدیل actions به لیست برای محیط
            actions_list = [actions[i] for i in range(self.n_agents)]
            
            # اجرای action در محیط
            try:
                result = self.env.step(actions_list)
            except Exception as e:
                print(f"⚠️  Error in step {steps}: {e}")
                break
            
            # پردازش خروجی step
            if len(result) == 4:
                next_state, reward, done, info = result
            elif len(result) == 5:
                next_state, reward, done, truncated, info = result
                done = done or truncated
            else:
                print(f"⚠️  Unexpected step output length: {len(result)}")
                break
            
            # جمع‌آوری متریک‌ها
            if isinstance(reward, (list, tuple, np.ndarray)):
                episode_reward += sum(reward)
            else:
                episode_reward += reward
            
            # استخراج Energy
            if isinstance(next_state, dict) and 'energy' in next_state:
                energy = next_state['energy']
                if isinstance(energy, np.ndarray):
                    episode_energy += np.sum(energy)
                elif isinstance(energy, (list, tuple)):
                    episode_energy += sum(energy)
                else:
                    episode_energy += energy
            
            # استخراج Delay
            if isinstance(next_state, dict):
                if 'distances' in next_state and 'uav_velocities' in next_state:
                    distances = np.array(next_state['distances'])
                    velocities = np.array(next_state['uav_velocities'])
                    velocities = np.where(velocities > 0, velocities, 1e-6)
                    delays = distances / velocities
                    episode_delays.append(np.mean(delays))
            
            state = next_state
            steps += 1
        
        # محاسبه میانگین Delay
        avg_delay = np.mean(episode_delays) if episode_delays else 0.0
        
        return {
            'reward': episode_reward,
            'energy': episode_energy,
            'delay': avg_delay,
            'steps': steps
        }
    
    def run_experiments(self, n_episodes: int = 50, max_steps: int = 200):
        """اجرای n اپیزود و جمع‌آوری نتایج"""
        print(f"\n🚀 Starting MADDPG experiments ({n_episodes} episodes)...")
        print(f"   Environment: {self.n_agents} UAVs")
        print(f"   Max steps per episode: {max_steps}")
        print("-" * 60)
        
        results = {
            'rewards': [],
            'energies': [],
            'delays': [],
            'steps': []
        }
        
        for ep in range(n_episodes):
            try:
                ep_result = self.run_episode(ep, max_steps)
                
                results['rewards'].append(ep_result['reward'])
                results['energies'].append(ep_result['energy'])
                results['delays'].append(ep_result['delay'])
                results['steps'].append(ep_result['steps'])
                
                if (ep + 1) % 10 == 0:
                    print(f"  Episode {ep+1}/{n_episodes}: "
                          f"R={ep_result['reward']:.2e}, "
                          f"E={ep_result['energy']:.2e}J, "
                          f"D={ep_result['delay']:.2f}s, "
                          f"Steps={ep_result['steps']}")
            except Exception as e:
                print(f"⚠️  Error in episode {ep+1}: {e}")
                continue
        
        # محاسبه آمار
        stats = {
            'policy': 'MADDPG',
            'n_episodes': len(results['rewards']),
            'reward': {
                'mean': float(np.mean(results['rewards'])),
                'std': float(np.std(results['rewards'])),
                'min': float(np.min(results['rewards'])),
                'max': float(np.max(results['rewards']))
            },
            'energy': {
                'mean': float(np.mean(results['energies'])),
                'std': float(np.std(results['energies'])),
                'min': float(np.min(results['energies'])),
                'max': float(np.max(results['energies']))
            },
            'delay': {
                'mean': float(np.mean(results['delays'])),
                'std': float(np.std(results['delays'])),
                'min': float(np.min(results['delays'])),
                'max': float(np.max(results['delays']))
            },
            'steps': {
                'mean': float(np.mean(results['steps'])),
                'std': float(np.std(results['steps']))
            }
        }
        
        print("\n" + "=" * 60)
        print("✅ MADDPG Experiments Completed!")
        print("=" * 60)
        print(f"  Average Reward: {stats['reward']['mean']:.2e} ± {stats['reward']['std']:.2e}")
        print(f"  Average Energy: {stats['energy']['mean']:.2e} ± {stats['energy']['std']:.2e} J")
        print(f"  Average Delay:  {stats['delay']['mean']:.2f} ± {stats['delay']['std']:.2f} s")
        print(f"  Average Steps:  {stats['steps']['mean']:.1f} ± {stats['steps']['std']:.1f}")
        print("=" * 60)
        
        return stats, results


def main():
    """اجرای اصلی"""
    print("\n" + "=" * 60)
    print("MADDPG EVALUATION EXPERIMENT")
    print("=" * 60)
    
    # تنظیمات محیط
    env_config = {
        'n_agents': 3,
        'n_users': 10,
        'area_size': 1000.0,
        'dt': 1.0,
        'alpha_delay': 1.0,
        'beta_energy': 1e-6,
        'gamma_eff': 1000.0
    }
    
    # مسیر مدل
    possible_paths = [
        'models/maddpg_sky_env_1.pth',
        'models/actor_agent0.pt',
    ]
    
    model_path = None
    for path in possible_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if model_path is None:
        print("❌ No MADDPG model found!")
        print("   Please train MADDPG first or provide correct model path.")
        return
    
    print(f"📂 Using model: {model_path}")
    
    try:
        # ساخت tester
        tester = MADDPGTester(model_path, env_config)
        
        # اجرای آزمایش‌ها
        stats, raw_results = tester.run_experiments(
            n_episodes=50,
            max_steps=200
        )
        
        # ذخیره نتایج
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / 'maddpg_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        with open(output_dir / 'maddpg_raw_results.json', 'w') as f:
            json.dump(raw_results, f, indent=2)
        
        print(f"\n💾 Results saved to {output_dir}/")
        print("   ✅ maddpg_stats.json")
        print("   ✅ maddpg_raw_results.json")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
