"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧪 اسکریپت جامع اجرای آزمایش‌های مقایسه موانع
مسیر: scripts/run_obstacle_experiments.py (NEW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, List
import argparse

from core.env_multi import MultiUAVEnvironment
from agents.agent_maddpg_multi import MADDPG_Agent
from agents.dqn import DQNAgent
from agents.bls import BLSAgent
from agents.ga import GAAgent
from agents.ecori import ECORIAgent
from analysis.realtime.obstacle_comparison import ObstacleComparison


class ObstacleExperimentRunner:
    """
    اجرای کننده آزمایش‌های جامع مقایسه موانع
    """
    
    def __init__(self,
                 n_episodes: int = 100,
                 max_steps: int = 500,
                 n_runs: int = 5,
                 save_dir: str = 'results/obstacle_experiments'):
        """
        Args:
            n_episodes: تعداد اپیزودها برای هر آزمایش
            max_steps: حداکثر گام در هر اپیزود
            n_runs: تعداد اجراهای مستقل (برای میانگین‌گیری)
            save_dir: مسیر ذخیره نتایج
        """
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.n_runs = n_runs
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.complexities = ['simple', 'medium', 'complex']
        self.algorithms = ['MADDPG', 'DQN', 'BLS', 'GA', 'ECORI']
        self.layers = ['Ground', 'Local', 'Edge', 'Cloud']
        
        self.comparison = ObstacleComparison()
        
        print("━" * 70)
        print("🧪 Obstacle Experiment Runner Initialized")
        print(f"   Episodes: {n_episodes}")
        print(f"   Max Steps: {max_steps}")
        print(f"   Runs: {n_runs}")
        print(f"   Complexities: {self.complexities}")
        print(f"   Algorithms: {self.algorithms}")
        print("━" * 70)
    
    def create_agent(self, 
                     algorithm: str, 
                     env: MultiUAVEnvironment,
                     layer: str) -> object:
        """
        ایجاد agent بر اساس الگوریتم
        """
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        if algorithm == 'MADDPG':
            return MADDPGAgent(
                n_agents=env.n_uavs,
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=256,
                lr_actor=1e-4,
                lr_critic=1e-3,
                gamma=0.99,
                tau=0.01
            )
        elif algorithm == 'DQN':
            return DQNAgent(
                state_dim=obs_dim,
                action_dim=act_dim,
                hidden_dim=128,
                lr=1e-3,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995
            )
        elif algorithm == 'BLS':
            return BLSAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                n_nodes=1000,
                n_features=200
            )
        elif algorithm == 'GA':
            return GAAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                population_size=50,
                mutation_rate=0.1,
                crossover_rate=0.7
            )
        elif algorithm == 'ECORI':
            return ECORIAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=256,
                lr=1e-3
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def run_single_experiment(self,
                             complexity: str,
                             algorithm: str,
                             layer: str,
                             run_id: int,
                             seed: int) -> Dict:
        """
        اجرای یک آزمایش واحد
        
        Returns:
            dict با متریک‌های عملکرد
        """
        # ایجاد محیط
        env = MultiUAVEnvironment(
            n_uavs=3,
            n_users=10,
            obstacle_complexity=complexity,
            enable_obstacles=True,
            seed=seed
        )
        
        # ایجاد agent
        agent = self.create_agent(algorithm, env, layer)
        
        # متریک‌های جمع‌آوری
        episode_rewards = []
        episode_delays = []
        episode_energies = []
        episode_collisions = []
        episode_success = []
        
        # حلقه آموزش
        pbar = tqdm(range(self.n_episodes), 
                   desc=f"{complexity}-{algorithm}-{layer}-Run{run_id}",
                   leave=False)
        
        for episode in pbar:
            obs, info = env.reset(seed=seed + episode)
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < self.max_steps:
                # انتخاب اقدام
                if algorithm == 'MADDPG':
                    actions = {}
                    for agent_id in range(env.n_uavs):
                        action = agent.select_action(obs[agent_id], agent_id)
                        actions[agent_id] = action
                else:
                    # سایر الگوریتم‌ها (فرض: single agent)
                    action = agent.select_action(obs[0])
                    actions = {i: action for i in range(env.n_uavs)}
                
                # اجرای گام
                next_obs, rewards, done, truncated, info = env.step(actions)
                
                # ذخیره تجربه (اختیاری)
                if algorithm in ['MADDPG', 'DQN']:
                    for agent_id in range(env.n_uavs):
                        agent.store_transition(
                            obs[agent_id],
                            actions[agent_id],
                            rewards[agent_id],
                            next_obs[agent_id],
                            done
                        )
                
                episode_reward += sum(rewards.values())
                obs = next_obs
                step += 1
            
            # آموزش (اختیاری)
            if algorithm in ['MADDPG', 'DQN'] and episode % 10 == 0:
                agent.train()
            
            # ذخیره متریک‌ها
            metrics = env.get_metrics()
            
            episode_rewards.append(episode_reward)
            episode_delays.append(metrics.get('avg_delay', 0))
            episode_energies.append(metrics.get('avg_energy', 0))
            episode_collisions.append(metrics.get('collision_rate', 0) * 100)
            episode_success.append(metrics.get('safety_rate', 0) * 100)
            
            # به‌روزرسانی progress bar
            pbar.set_postfix({
                'Reward': f"{episode_reward:.2f}",
                'Collision': f"{episode_collisions[-1]:.1f}%"
            })
        
        pbar.close()
        
        # محاسبه میانگین (10 اپیزود آخر)
        last_n = 10
        
        return {
            'avg_delay': np.mean(episode_delays[-last_n:]),
            'avg_energy': np.mean(episode_energies[-last_n:]),
            'avg_reward': np.mean(episode_rewards[-last_n:]),
            'collision_rate': np.mean(episode_collisions[-last_n:]),
            'success_rate': np.mean(episode_success[-last_n:]),
            'path_length': 0,  # TODO: محاسبه از env
            'computation_time': 0,  # TODO: اندازه‌گیری زمان
            'safety_score': 100 - np.mean(episode_collisions[-last_n:])
        }
    
    def run_all_experiments(self):
        """
        اجرای تمام ترکیبات آزمایش
        """
        total_experiments = (
            len(self.complexities) * 
            len(self.algorithms) * 
            len(self.layers) * 
            self.n_runs
        )
        
        print(f"\n🚀 Starting {total_experiments} experiments...")
        print("━" * 70)
        
        experiment_id = 0
        start_time = datetime.now()
        
        for complexity in self.complexities:
            print(f"\n{'='*70}")
            print(f"📊 Complexity Level: {complexity.upper()}")
            print(f"{'='*70}")
            
            for algorithm in self.algorithms:
                print(f"\n  🤖 Algorithm: {algorithm}")
                
                for layer in self.layers:
                    print(f"    📍 Layer: {layer}")
                    
                    # اجرای چند run برای میانگین‌گیری
                    run_results = []
                    
                    for run_id in range(self.n_runs):
                        seed = 42 + experiment_id
                        
                        try:
                            result = self.run_single_experiment(
                                complexity=complexity,
                                algorithm=algorithm,
                                layer=layer,
                                run_id=run_id,
                                seed=seed
                            )
                            run_results.append(result)
                            
                            print(f"      ✅ Run {run_id+1}/{self.n_runs} completed")
                            
                        except Exception as e:
                            print(f"      ❌ Run {run_id+1} failed: {str(e)}")
                            continue
                        
                        experiment_id += 1
                    
                    # میانگین‌گیری نتایج
                    if run_results:
                        avg_result = {
                            key: np.mean([r[key] for r in run_results])
                            for key in run_results[0].keys()
                        }
                        
                        # اضافه کردن به comparison
                        self.comparison.add_result(
                            complexity=complexity,
                            algorithm=algorithm,
                            layer=layer,
                            metrics=avg_result
                        )
                        
                        print(f"      📈 Avg Results: "
                              f"Delay={avg_result['avg_delay']:.2f}ms, "
                              f"Collision={avg_result['collision_rate']:.1f}%")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "━" * 70)
        print(f"✅ All experiments completed!")
        print(f"⏱️  Total time: {duration/60:.1f} minutes")
        print("━" * 70)
    
    def generate_reports(self):
        """
        تولید گزارش‌ها و نمودارها
        """
        print("\n🎨 Generating reports...")
        
        # ذخیره نتایج خام
        self.comparison.save_results(
            filename='obstacle_comparison_results.json'
        )
        print("  ✅ Raw results saved")
        
        # تولید تحلیل‌های مختلف
        try:
            # 1. مقایسه داخلی
            for complexity in self.complexities:
                df = self.comparison.generate_intra_complexity_comparison(complexity)
                print(f"  ✅ Intra-complexity analysis: {complexity}")
            
            # 2. مقایسه بین‌لایه‌ای
            for algo in ['MADDPG', 'DQN']:
                for complexity in ['simple', 'complex']:
                    df = self.comparison.generate_inter_layer_comparison(complexity, algo)
                    print(f"  ✅ Inter-layer analysis: {algo} ({complexity})")
            
            # 3. مقایسه متقاطع
            for algo in ['MADDPG', 'BLS']:
                for layer in ['Edge', 'Cloud']:
                    df = self.comparison.generate_cross_complexity_comparison(algo, layer)
                    print(f"  ✅ Cross-complexity analysis: {algo} on {layer}")
            
            # 4. Heatmap
            self.comparison.generate_heatmap_comparison()
            print("  ✅ Heatmap generated")
            
            # 5. جدول خلاصه
            summary_df = self.comparison.generate_summary_table()
            print("  ✅ Summary table generated")
            
            print("\n📁 All reports saved to: results/")
            
        except Exception as e:
            print(f"  ❌ Error generating reports: {str(e)}")
    
    def save_metadata(self):
        """ذخیره اطلاعات آزمایش"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'n_episodes': self.n_episodes,
            'max_steps': self.max_steps,
            'n_runs': self.n_runs,
            'complexities': self.complexities,
            'algorithms': self.algorithms,
            'layers': self.layers,
            'total_experiments': (
                len(self.complexities) * 
                len(self.algorithms) * 
                len(self.layers) * 
                self.n_runs
            )
        }
        
        with open(f'{self.save_dir}/experiment_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Metadata saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 اجرای اصلی
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description='🧪 SkyMind Obstacle Comparison Experiments'
    )
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes per experiment')
    parser.add_argument('--steps', type=int, default=500,
                       help='Max steps per episode')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of independent runs')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (10 episodes, 1 run)')
    
    args = parser.parse_args()
    
    if args.quick:
        print("⚡ Quick test mode enabled")
        args.episodes = 10
        args.runs = 1
    
    # ایجاد runner
    runner = ObstacleExperimentRunner(
        n_episodes=args.episodes,
        max_steps=args.steps,
        n_runs=args.runs
    )
    
    # اجرای آزمایش‌ها
    runner.run_all_experiments()
    
    # تولید گزارش‌ها
    runner.generate_reports()
    
    # ذخیره metadata
    runner.save_metadata()
    
    print("\n" + "🎉" * 35)
    print("All done! Check results/ directory for outputs.")
    print("🎉" * 35)


if __name__ == "__main__":
    main()
