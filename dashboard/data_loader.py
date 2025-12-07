# dashboard/data_loader.py
import json
import numpy as np
from pathlib import Path

class TrainingDataLoader:
    """بارگذاری داده‌های آموزش از فایل‌های JSON"""
    
    def __init__(self, base_path='models'):
        self.base_path = Path(base_path)
        self.episodes = []
        self.rewards = []
        self.losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.agent_names = []
    
    def load_level_data(self, level):
        """بارگذاری داده‌های یک سطح خاص"""
        level_map = {
            'level1': 'maddpg',              # ✅ اصلاح شد - اشاره به فایل 4000 اپیزودی
            'level2': 'level2_medium',
            'level3': 'level3_complex'
        }
        
        folder_name = level_map.get(level, level)
        json_path = self.base_path / folder_name / 'training_history.json'
        
        print(f"📂 Loading from: {json_path}")
        
        if not json_path.exists():
            print(f"⚠️ فایل {json_path} یافت نشد!")
            return None
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} episodes from {json_path}")
        
        # ✅ بررسی ساختار JSON (فرمت کلید-مقدار یا لیستی)
        # فرمت 1: {'1': {...}, '2': {...}, ...}  ← فرمت جدید از train_curriculum_learning.py
        # فرمت 2: {'episodes': [...], 'rewards': [...]}  ← فرمت قدیمی
        
        if isinstance(data, dict) and all(key.isdigit() for key in list(data.keys())[:5]):
            # ✅ فرمت جدید: هر کلید شماره episode است
            episodes = []
            rewards_per_episode = []
            actor_losses = []
            critic_losses = []
            rewards_agent0 = []
            rewards_agent1 = []
            noise_std = []
            
            for ep_num in sorted(data.keys(), key=int):
                ep_data = data[ep_num]
                
                # ✅ اصلاح اصلی: استفاده از 'avg_reward' به جای 'reward'
                avg_reward = ep_data.get('avg_reward', 0)
                
                episodes.append(ep_data.get('episode', int(ep_num)))
                rewards_per_episode.append(avg_reward)
                critic_losses.append(ep_data.get('critic_loss', 0))
                actor_losses.append(ep_data.get('actor_loss', 0))
                noise_std.append(ep_data.get('noise_std', 0))
                
                # ✅ استخراج پاداش هر agent
                rewards_dict = ep_data.get('rewards', {})
                if isinstance(rewards_dict, dict):
                    rewards_agent0.append(rewards_dict.get('agent_0', avg_reward))
                    rewards_agent1.append(rewards_dict.get('agent_1', avg_reward))
                else:
                    # اگر rewards دیکشنری نبود، از avg_reward استفاده کن
                    rewards_agent0.append(avg_reward)
                    rewards_agent1.append(avg_reward)
            
        else:
            # ✅ فرمت قدیمی
            episodes = data.get('episodes', [])
            raw_rewards = data.get('rewards', [])
            actor_losses = data.get('actor_losses', [])
            critic_losses = data.get('critic_losses', [])
            
            # تبدیل rewards بر اساس ساختار
            if raw_rewards and isinstance(raw_rewards[0], list):
                rewards_per_episode = [sum(ep_rewards)/len(ep_rewards) if ep_rewards else 0 
                                       for ep_rewards in raw_rewards]
                rewards_agent0 = [ep[0] if len(ep) > 0 else 0 for ep in raw_rewards]
                rewards_agent1 = [ep[1] if len(ep) > 1 else 0 for ep in raw_rewards]
            else:
                rewards_per_episode = raw_rewards
                rewards_agent0 = raw_rewards
                rewards_agent1 = []
            
            noise_std = data.get('noise_std', [])
        
        # ذخیره در متغیرهای کلاس
        self.episodes = episodes if episodes else list(range(1, len(rewards_per_episode) + 1))
        self.rewards = rewards_per_episode
        self.actor_losses = actor_losses
        self.critic_losses = critic_losses
        self.losses = actor_losses + critic_losses
        self.agent_names = ['agent_0', 'agent_1']
        
        print(f"📊 Parsed data: {len(self.episodes)} episodes, avg_reward={np.mean(self.rewards):.2f}")
        
        return {
            'episodes': self.episodes,
            'rewards_agent0': rewards_agent0,
            'rewards_agent1': rewards_agent1,
            'actor_loss_agent0': actor_losses,
            'critic_loss_agent0': critic_losses,
            'noise_std': noise_std
        }
    
    def get_summary_stats(self):
        """خلاصه آمار کلی تریینگ"""
        try:
            if len(self.rewards) == 0:
                return self._empty_stats()
            
            # ✅ محاسبه نرخ موفقیت: اپیزودهایی با reward بیشتر از -10
            successful_episodes = sum(1 for r in self.rewards if r > -10)
            success_rate = (successful_episodes / len(self.rewards) * 100)
            
            # محاسبه میانگین reward
            avg_reward = float(np.mean(self.rewards))
            
            # آخرین loss
            latest_loss = float(self.losses[-1]) if len(self.losses) > 0 else 0.0
            
            # میانگین critic loss
            avg_critic_loss = float(np.mean(self.critic_losses)) if len(self.critic_losses) > 0 else 0.0
            
            # میانگین actor loss
            avg_actor_loss = float(np.mean(self.actor_losses)) if len(self.actor_losses) > 0 else 0.0
            
            print(f"📈 Summary: episodes={len(self.episodes)}, avg_reward={avg_reward:.2f}, success_rate={success_rate:.1f}%")
            
            return {
                'total_episodes': len(self.episodes),
                'total_agents': len(self.agent_names),
                'avg_reward': avg_reward,
                'latest_loss': latest_loss,
                'success_rate': success_rate,
                'avg_critic_loss': avg_critic_loss,
                'avg_actor_loss': avg_actor_loss
            }
        except Exception as e:
            print(f"⚠️ Error in get_summary_stats: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_stats()
    
    def _empty_stats(self):
        """بازگشت آمار خالی در صورت خطا"""
        return {
            'total_episodes': 0,
            'total_agents': 0,
            'avg_reward': 0.0,
            'latest_loss': 0.0,
            'success_rate': 0.0,
            'avg_critic_loss': 0.0,
            'avg_actor_loss': 0.0
        }
