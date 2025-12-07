import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class MetricsExtractor:
    """Extract training metrics from log files"""
    
    def __init__(self, log_file_path):
        self.log_file = log_file_path
        self.episodes = []
        self.rewards = []
        self.losses = []
        self.epsilons = []
        
    def extract_episode_metrics(self):
        """استخراج Episode, Reward, Loss از لاگ"""
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # پیدا کردن خطوط Episode
                episode_match = re.search(r'Episode (\d+)', line)
                if episode_match:
                    episode_num = int(episode_match.group(1))
                    
                    # استخراج Reward
                    reward_match = re.search(r'Reward[:\s]+(-?\d+\.?\d*)', line)
                    if reward_match:
                        reward = float(reward_match.group(1))
                        self.episodes.append(episode_num)
                        self.rewards.append(reward)
                    
                    # استخراج Loss
                    loss_match = re.search(r'Loss[:\s]+(\d+\.?\d*)', line)
                    if loss_match:
                        loss = float(loss_match.group(1))
                        self.losses.append(loss)
                    
                    # استخراج Epsilon
                    eps_match = re.search(r'Epsilon[:\s]+(\d+\.?\d*)', line)
                    if eps_match:
                        epsilon = float(eps_match.group(1))
                        self.epsilons.append(epsilon)
        
        return self.create_dataframe()
    
    def create_dataframe(self):
        """ساخت DataFrame از داده‌های استخراج شده"""
        
        # پر کردن داده‌های ناقص
        max_len = len(self.episodes)
        
        if len(self.losses) < max_len:
            self.losses.extend([None] * (max_len - len(self.losses)))
        if len(self.epsilons) < max_len:
            self.epsilons.extend([None] * (max_len - len(self.epsilons)))
        
        df = pd.DataFrame({
            'Episode': self.episodes,
            'Reward': self.rewards,
            'Loss': self.losses[:max_len],
            'Epsilon': self.epsilons[:max_len]
        })
        
        return df
    
    def plot_metrics(self, df, save_path='metrics_plot.png'):
        """رسم نمودارهای Metrics"""
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot Reward
        axes[0].plot(df['Episode'], df['Reward'], 'b-', linewidth=0.5, alpha=0.3)
        axes[0].plot(df['Episode'], df['Reward'].rolling(100).mean(), 'r-', linewidth=2, label='Moving Avg (100)')
        axes[0].set_ylabel('Mean Reward')
        axes[0].set_title('Training Reward Over Episodes')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot Loss
        if df['Loss'].notna().any():
            axes[1].plot(df['Episode'], df['Loss'], 'g-', linewidth=0.5, alpha=0.3)
            axes[1].plot(df['Episode'], df['Loss'].rolling(100).mean(), 'r-', linewidth=2, label='Moving Avg (100)')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Loss Over Episodes')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Plot Epsilon
        if df['Epsilon'].notna().any():
            axes[2].plot(df['Episode'], df['Epsilon'], 'orange', linewidth=1)
            axes[2].set_ylabel('Epsilon')
            axes[2].set_xlabel('Episode')
            axes[2].set_title('Exploration Rate (Epsilon) Over Episodes')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ نمودار ذخیره شد: {save_path}")
        
    def analyze_recent_trend(self, df, window=100):
        """تحلیل روند اخیر"""
        
        recent_df = df.tail(window)
        
        analysis = {
            'Last_Episode': recent_df['Episode'].iloc[-1],
            'Mean_Reward_Recent': recent_df['Reward'].mean(),
            'Reward_Trend': 'Increasing' if recent_df['Reward'].is_monotonic_increasing else 'Decreasing/Flat',
            'Reward_Std': recent_df['Reward'].std(),
        }
        
        if recent_df['Loss'].notna().any():
            analysis['Mean_Loss_Recent'] = recent_df['Loss'].mean()
            analysis['Loss_Trend'] = 'Decreasing' if recent_df['Loss'].is_monotonic_decreasing else 'Increasing/Flat'
        
        if recent_df['Epsilon'].notna().any():
            analysis['Current_Epsilon'] = recent_df['Epsilon'].iloc[-1]
        
        return analysis


# استفاده:
def main():
    # مسیر فایل لاگ
    log_path = "training_log.txt"  # یا هر مسیری که لاگ‌ها رو داری
    
    print("🔍 شروع استخراج Metrics...")
    extractor = MetricsExtractor(log_path)
    
    # استخراج داده‌ها
    df = extractor.extract_episode_metrics()
    
    if df.empty:
        print("❌ هیچ داده‌ای پیدا نشد! فرمت لاگ رو چک کن.")
        return
    
    print(f"✅ {len(df)} Episode استخراج شد!")
    
    # نمایش خلاصه
    print("\n📊 خلاصه داده‌ها:")
    print(df.describe())
    
    # تحلیل روند اخیر
    print("\n📈 تحلیل 100 Episode اخیر:")
    analysis = extractor.analyze_recent_trend(df)
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # رسم نمودارها
    extractor.plot_metrics(df)
    
    # ذخیره CSV
    csv_path = "training_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ داده‌ها ذخیره شدند: {csv_path}")

if __name__ == "__main__":
    main()
