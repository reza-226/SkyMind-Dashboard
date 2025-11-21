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
            'level1': 'level1_simple',
            'level2': 'level2_medium',
            'level3': 'level3_complex'
        }
        
        folder_name = level_map.get(level, level)
        json_path = self.base_path / folder_name / 'training_history.json'
        
        if not json_path.exists():
            print(f"⚠️ فایل {json_path} یافت نشد!")
            return None
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # ✅ استفاده از کلیدهای واقعی JSON
        episodes = data.get('episodes', [])
        raw_rewards = data.get('rewards', [])
        actor_losses = data.get('actor_losses', [])
        critic_losses = data.get('critic_losses', [])
        
        # ✅ تبدیل rewards بر اساس ساختار
        # اگر rewards لیست تو در تو باشه (مثلاً [[r1_a0, r1_a1], [r2_a0, r2_a1], ...])
        if raw_rewards and isinstance(raw_rewards[0], list):
            # محاسبه میانگین reward همه agent ها در هر episode
            rewards_per_episode = [sum(ep_rewards)/len(ep_rewards) if ep_rewards else 0 
                                   for ep_rewards in raw_rewards]
            # برای agent 0 و agent 1 (فرض می‌کنیم 2 agent داریم)
            rewards_agent0 = [ep[0] if len(ep) > 0 else 0 for ep in raw_rewards]
            rewards_agent1 = [ep[1] if len(ep) > 1 else 0 for ep in raw_rewards]
        else:
            # اگر یک لیست ساده بود
            rewards_per_episode = raw_rewards
            rewards_agent0 = raw_rewards
            rewards_agent1 = []
        
        # ذخیره در متغیرهای کلاس برای استفاده در get_summary_stats
        self.episodes = episodes if episodes else list(range(1, len(raw_rewards) + 1))
        self.rewards = rewards_per_episode
        self.actor_losses = actor_losses
        self.critic_losses = critic_losses
        self.losses = actor_losses + critic_losses
        self.agent_names = ['agent_0', 'agent_1']
        
        return {
            'episodes': self.episodes,
            'rewards_agent0': rewards_agent0,
            'rewards_agent1': rewards_agent1,
            'actor_loss_agent0': actor_losses,
            'critic_loss_agent0': critic_losses,
            'noise_std': data.get('noise_std', [])
        }
    
    def get_summary_stats(self):
        """خلاصه آمار کلی تریینگ"""
        try:
            if len(self.rewards) == 0:
                return self._empty_stats()
            
            # ✅ محاسبه نرخ موفقیت بر اساس بهبود نسبت به میانگین
            # روش 1: Episode موفق = reward بالاتر از میانگین کل
            avg_reward_all = np.mean(self.rewards)
            successful_episodes = sum(1 for r in self.rewards if r > avg_reward_all)
            
            # روش 2 (پیشنهادی برای پروژه Offloading):
            # Episode موفق = reward بالاتر از threshold مشخص
            # مثلاً اگه reward شما بین -50 تا +10 باشه، threshold = -10
            # SUCCESS_THRESHOLD = -10.0
            # successful_episodes = sum(1 for r in self.rewards if r > SUCCESS_THRESHOLD)
            
            # روش 3 (برای بهبود پیوسته):
            # نرخ موفقیت = درصد episode هایی که reward بهتر از نیمه اول آموزش بوده
            # mid_point = len(self.rewards) // 2
            # if mid_point > 0:
            #     avg_first_half = np.mean(self.rewards[:mid_point])
            #     successful_episodes = sum(1 for r in self.rewards[mid_point:] if r > avg_first_half)
            #     success_rate = (successful_episodes / (len(self.rewards) - mid_point) * 100)
            # else:
            #     success_rate = 0.0
            
            success_rate = (successful_episodes / len(self.rewards) * 100)
            
            # محاسبه میانگین reward
            avg_reward = float(np.mean(self.rewards))
            
            # آخرین loss
            latest_loss = float(self.losses[-1]) if len(self.losses) > 0 else 0.0
            
            # میانگین critic loss
            avg_critic_loss = float(np.mean(self.critic_losses)) if len(self.critic_losses) > 0 else 0.0
            
            # میانگین actor loss
            avg_actor_loss = float(np.mean(self.actor_losses)) if len(self.actor_losses) > 0 else 0.0
            
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
