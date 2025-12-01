"""
اسکریپت اجرای مطالعات Ablation
مسیر: scripts/run_ablation_study.py
"""

import sys
import os
from pathlib import Path

# اضافه کردن مسیر پروژه
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
import time
from datetime import datetime
import torch

# Import محیط
from pettingzoo.mpe import simple_tag_v3

# Import variants
from core.evaluation.ablation_variants import (
    FullMADDPGVariant,
    NoGATVariant,
    NoTemporalVariant,
    DecentralizedVariant,
    SimplerArchVariant
)


class AblationStudyRunner:
    """مدیریت اجرای مطالعات Ablation"""
    
    def __init__(self, results_dir="results/ablation"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # محیط
        self.env = None
        self.num_agents = 3
        
        # تنظیمات آموزش
        self.train_config = {
            "num_episodes": 500,
            "max_steps_per_episode": 100,
            "eval_interval": 50,
            "eval_episodes": 20,
            "gamma": 0.95,
            "tau": 0.001,
            "batch_size": 64,
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "buffer_size": 100000
        }
        
        # Variants برای آزمایش
        self.variants = {
            "full_model": FullMADDPGVariant,
            "no_gat": NoGATVariant,
            "no_temporal": NoTemporalVariant,
            "decentralized": DecentralizedVariant,
            "simpler_arch": SimplerArchVariant
        }
        
        # نتایج
        self.results = {}
    
    def create_env(self):
        """ایجاد محیط"""
        env = simple_tag_v3.parallel_env(
            num_good=1,
            num_adversaries=2,
            num_obstacles=2,
            max_cycles=self.train_config["max_steps_per_episode"],
            continuous_actions=True
        )
        return env
    
    def get_obs_action_dims(self):
        """✅ محاسبه دقیق ابعاد برای همه Agents"""
        env = self.create_env()
        obs_dict, _ = env.reset()
        
        print("\n" + "="*60)
        print("🔍 بررسی دقیق ابعاد برای همه Agents...")
        print("="*60)
        
        # دیکشنری برای ذخیره ابعاد هر Agent
        obs_dims = {}
        action_dims = {}
        
        # ✅ اجرای چند step برای دریافت تمام observations
        all_obs_samples = {agent: [] for agent in env.agents}
        
        for step_idx in range(10):
            actions = {agent: env.action_space(agent).sample() 
                      for agent in env.agents}
            obs_dict, _, terminations, truncations, _ = env.step(actions)
            
            # جمع‌آوری نمونه‌ها
            for agent in env.agents:
                if agent in obs_dict:
                    obs = obs_dict[agent]
                    obs_dim = obs.shape[0] if hasattr(obs, 'shape') else len(obs)
                    all_obs_samples[agent].append(obs_dim)
            
            # اگر همه agents تمام شدند، محیط را reset کن
            if not env.agents:
                env.reset()
        
        # محاسبه ابعاد برای هر Agent
        for agent in all_obs_samples:
            if all_obs_samples[agent]:
                obs_dims[agent] = max(all_obs_samples[agent])
                action_space = env.action_space(agent)
                action_dims[agent] = action_space.shape[0]
                
                print(f"\n   Agent: {agent}")
                print(f"      Obs dims مشاهده شده: {all_obs_samples[agent]}")
                print(f"      Max Obs dim: {obs_dims[agent]}")
                print(f"      Action dim: {action_dims[agent]}")
                print(f"      Action range: [{action_space.low[0]:.2f}, {action_space.high[0]:.2f}]")
        
        # استفاده از بیشترین مقدار برای همه
        obs_dim = max(obs_dims.values()) if obs_dims else 14
        action_dim = max(action_dims.values()) if action_dims else 5
        
        env.close()
        
        print(f"\n{'='*60}")
        print(f"📊 ابعاد نهایی انتخاب شده:")
        print(f"   - Max Observation dim: {obs_dim}")
        print(f"   - Max Action dim: {action_dim}")
        print(f"   - Number of agents: {len(obs_dims)}")
        print(f"{'='*60}\n")
        
        return obs_dim, action_dim
    
    def normalize_observation(self, obs, target_dim):
        """✅ نرمال‌سازی و تنظیم ابعاد observation"""
        current_dim = len(obs)
        
        if current_dim < target_dim:
            # Pad با صفر
            obs = np.pad(obs, (0, target_dim - current_dim), mode='constant', constant_values=0)
        elif current_dim > target_dim:
            # کوتاه کردن
            obs = obs[:target_dim]
        
        return obs
    
    def normalize_action(self, action):
        """✅ نرمال‌سازی دقیق action به بازه [0, 1]"""
        # Clip به بازه [0, 1]
        action = np.clip(action, 0.0, 1.0)
        
        # گرد کردن به 6 رقم اعشار برای دقت بیشتر
        action = np.round(action, decimals=6)
        
        # اطمینان از اینکه مقادیر حدی را کمی جابجا می‌کنیم
        epsilon = 1e-6
        action = np.where(action < epsilon, epsilon, action)
        action = np.where(action > (1.0 - epsilon), 1.0 - epsilon, action)
        
        return action
    
    def train_variant(self, variant_name, variant_class, obs_dim, action_dim):
        """آموزش یک variant"""
        
        print(f"\n{'='*60}")
        print(f"🚀 شروع آموزش: {variant_name}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # ایجاد مدل
        model = variant_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_agents=self.num_agents,
            **self.train_config
        )
        
        # ایجاد محیط
        env = self.create_env()
        
        # ذخیره‌سازی
        variant_dir = self.results_dir / variant_name
        variant_dir.mkdir(exist_ok=True)
        
        # متریک‌ها
        episode_rewards = []
        eval_rewards = []
        best_eval_reward = -float('inf')
        
        # شمارش هشدارهای clipping
        clipping_warnings = 0
        
        try:
            for episode in range(self.train_config["num_episodes"]):
                # Reset محیط
                obs_dict, info = env.reset()
                
                episode_reward = 0
                step = 0
                
                while env.agents:
                    # انتخاب actions برای همه agents
                    actions = {}
                    for agent_id in env.agents:
                        obs = obs_dict[agent_id]
                        
                        # ✅ نرمال‌سازی observation
                        obs = self.normalize_observation(obs, obs_dim)
                        
                        # دریافت action از مدل
                        action = model.select_action(agent_id, obs, add_noise=True)
                        
                        # ✅ نرمال‌سازی دقیق action
                        action = self.normalize_action(action)
                        
                        actions[agent_id] = action
                    
                    # اجرای step
                    next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
                    
                    # ذخیره transitions
                    for agent_id in env.agents:
                        if agent_id in obs_dict and agent_id in next_obs_dict:
                            obs = obs_dict[agent_id]
                            next_obs = next_obs_dict[agent_id]
                            
                            # ✅ نرمال‌سازی observations
                            obs = self.normalize_observation(obs, obs_dim)
                            next_obs = self.normalize_observation(next_obs, obs_dim)
                            
                            model.store_transition(
                                agent_id=agent_id,
                                state=obs,
                                action=actions[agent_id],
                                reward=rewards[agent_id],
                                next_state=next_obs,
                                done=terminations[agent_id] or truncations[agent_id]
                            )
                    
                    # آپدیت پاداش
                    episode_reward += sum(rewards.values())
                    
                    # آپدیت مدل
                    model.update()
                    
                    # آپدیت state
                    obs_dict = next_obs_dict
                    step += 1
                
                episode_rewards.append(episode_reward)
                
                # نمایش پیشرفت
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"Episode {episode+1}/{self.train_config['num_episodes']} | "
                          f"Avg Reward: {avg_reward:.2f}")
                
                # ارزیابی
                if (episode + 1) % self.train_config["eval_interval"] == 0:
                    eval_reward = self.evaluate_variant(model, env, obs_dim)
                    eval_rewards.append(eval_reward)
                    
                    print(f"📊 Evaluation at episode {episode+1}: {eval_reward:.2f}")
                    
                    # ذخیره بهترین مدل
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        model.save(variant_dir / "best_model.pt")
                        print(f"✅ New best model saved: {eval_reward:.2f}")
            
            # محاسبه زمان
            training_time = (time.time() - start_time) / 60
            
            # ذخیره نتایج
            results = {
                "variant": variant_name,
                "best_eval_reward": float(best_eval_reward),
                "final_avg_reward": float(np.mean(episode_rewards[-100:])),
                "training_time_minutes": float(training_time),
                "episode_rewards": [float(r) for r in episode_rewards],
                "eval_rewards": [float(r) for r in eval_rewards],
                "config": {
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "num_agents": self.num_agents
                }
            }
            
            with open(variant_dir / "training_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            self.results[variant_name] = results
            
            print(f"\n✅ آموزش {variant_name} با موفقیت انجام شد!")
            print(f"   Best Reward: {best_eval_reward:.2f}")
            print(f"   Final Avg: {results['final_avg_reward']:.2f}")
            print(f"   Time: {training_time:.1f} min")
            
        except Exception as e:
            print(f"\n❌ خطا در آموزش {variant_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self.results[variant_name] = {
                "variant": variant_name,
                "error": str(e),
                "status": "failed"
            }
        
        finally:
            env.close()
    
    def evaluate_variant(self, model, env, obs_dim):
        """ارزیابی مدل"""
        eval_rewards = []
        
        for _ in range(self.train_config["eval_episodes"]):
            obs_dict, _ = env.reset()
            episode_reward = 0
            
            while env.agents:
                actions = {}
                for agent_id in env.agents:
                    obs = obs_dict[agent_id]
                    
                    # ✅ نرمال‌سازی observation
                    obs = self.normalize_observation(obs, obs_dim)
                    
                    # دریافت action بدون نویز
                    action = model.select_action(agent_id, obs, add_noise=False)
                    
                    # ✅ نرمال‌سازی دقیق action
                    action = self.normalize_action(action)
                    
                    actions[agent_id] = action
                
                next_obs_dict, rewards, terminations, truncations, _ = env.step(actions)
                
                episode_reward += sum(rewards.values())
                obs_dict = next_obs_dict
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def run_all_variants(self):
        """اجرای تمام variants"""
        
        print("\n" + "="*60)
        print("🔬 شروع مطالعات Ablation")
        print("="*60)
        
        # ✅ محاسبه ابعاد واقعی از محیط
        obs_dim, action_dim = self.get_obs_action_dims()
        
        print(f"\n📊 تنظیمات آموزش:")
        print(f"   Observation dim: {obs_dim}")
        print(f"   Action dim: {action_dim}")
        print(f"   Number of agents: {self.num_agents}")
        print(f"   Episodes: {self.train_config['num_episodes']}")
        print(f"   Eval interval: {self.train_config['eval_interval']}")
        
        # آموزش هر variant
        for variant_name, variant_class in self.variants.items():
            self.train_variant(variant_name, variant_class, obs_dim, action_dim)
        
        # ذخیره خلاصه
        self.save_summary()
        
        # نمایش نتایج
        self.print_summary()
    
    def save_summary(self):
        """ذخیره خلاصه نتایج"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": self.train_config,
            "results": self.results
        }
        
        with open(self.results_dir / "ablation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    def print_summary(self):
        """نمایش خلاصه نتایج"""
        print("\n" + "="*60)
        print("✅ تمام آزمایش‌های Ablation تکمیل شد!")
        print("="*60)
        
        print("\n📊 خلاصه نتایج Ablation:\n")
        print(f"{'Variant':<20} {'Best Reward':<15} {'Final Avg':<15} {'Time (min)':<12}")
        print("-" * 62)
        
        for variant_name, result in self.results.items():
            if "error" in result:
                print(f"{variant_name:<20} {'FAILED':<15} {'-':<15} {'-':<12}")
            else:
                print(f"{variant_name:<20} "
                      f"{result['best_eval_reward']:<15.2f} "
                      f"{result['final_avg_reward']:<15.2f} "
                      f"{result['training_time_minutes']:<12.1f}")
        
        print("\n💾 نتایج ذخیره شده در:", self.results_dir)


def main():
    """اجرای اصلی"""
    runner = AblationStudyRunner(results_dir="results/ablation")
    runner.run_all_variants()


if __name__ == "__main__":
    main()
