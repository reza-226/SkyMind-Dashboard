"""
تست Agent با GNN و لاگر - نسخه صحیح نهایی
"""
import torch
import numpy as np
from core.drl_agent import MATOAgent
from core.training_logger import TrainingLogger
from utils.graph_utils import generate_random_dag


def test_agent_with_logging():
    """تست کامل Agent با GNN و Logger"""
    
    print("=" * 60)
    print("🧪 Testing MATO Agent with GNN + Logger")
    print("=" * 60)
    
    # تنظیمات - با پارامترهای صحیح
    config = {
        'agent': {
            'agent_id': 0,
            'state_dim': 10,
            'action_dim': 5,
            'hidden_dim': 256,
            'lr': 3e-4,
            'gamma': 0.99
        },
        'gnn': {
            'node_feature_dim': 9,
            'edge_feature_dim': 3,
            'hidden_dim': 256,
            'embedding_dim': 256,
            'num_gat_layers': 3,  # ✅ تصحیح شد
            'num_heads': 4,
            'dropout': 0.1
        }
    }
    
    # ساخت Logger
    logger = TrainingLogger(log_dir="logs/test_run")
    logger.save_config(config)
    
    # ساخت Agent
    agent = MATOAgent(
        agent_id=config['agent']['agent_id'],
        state_dim=config['agent']['state_dim'],
        action_dim=config['agent']['action_dim'],
        gnn_config=config['gnn'],
        hidden_dim=config['agent']['hidden_dim'],
        lr=config['agent']['lr'],
        gamma=config['agent']['gamma']
    )
    
    print("\n✅ Agent created successfully")
    print(f"   - State dim: {config['agent']['state_dim']}")
    print(f"   - Action dim: {config['agent']['action_dim']}")
    print(f"   - GNN hidden dim: {config['gnn']['hidden_dim']}")
    print(f"   - GNN embedding dim: {config['gnn']['embedding_dim']}")
    print(f"   - GNN GAT layers: {config['gnn']['num_gat_layers']}")
    
    # شبیه‌سازی training loop
    num_episodes = 10
    steps_per_episode = 20
    
    for episode in range(num_episodes):
        agent.reset_episode()
        episode_reward = 0
        
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        
        for step in range(steps_per_episode):
            # تولید DAG تصادفی
            dag = generate_random_dag(
                num_tasks=np.random.randint(5, 15),
                edge_probability=0.3,
                seed=episode * 1000 + step
            )
            
            # تولید env state تصادفی
            env_state = np.random.randn(config['agent']['state_dim'])
            
            # انتخاب action
            action, info = agent.select_action(dag, env_state)
            
            # شبیه‌سازی reward
            reward = np.random.randn() * 10
            agent.store_reward(reward)
            episode_reward += reward
            
            # اضافه کردن اطلاعات برای لاگ
            info.update({
                'task_id': step,
                'offload_target': f"target_{action}",
                'delay': np.random.rand() * 100,
                'energy': np.random.rand() * 50,
                'num_nodes': len(dag.nodes),
                'num_edges': sum(len(task.successors) for task in dag.nodes.values()),
                'critical_path_length': np.random.randint(3, len(dag.nodes)),
                'graph_density': np.random.rand()
            })
            
            # لاگ step
            logger.log_step(episode, step, agent.agent_id, action, reward, info)
        
        # به‌روزرسانی policy
        losses = agent.update()
        
        # لاگ episode
        metrics = agent.get_metrics()
        metrics.update({
            'avg_reward': episode_reward / steps_per_episode,
            'episode_length': steps_per_episode,
            'success_rate': np.random.rand(),
            'avg_delay': np.random.rand() * 100,
            'avg_energy': np.random.rand() * 50,
            'num_offloaded_tasks': np.random.randint(5, 15)
        })
        metrics.update(losses)
        
        logger.log_episode(episode, episode_reward, metrics)
        
        print(f"   Reward: {episode_reward:.2f}")
        if losses:
            print(f"   Actor Loss: {losses.get('actor_loss', 0):.4f}")
            print(f"   Critic Loss: {losses.get('critic_loss', 0):.4f}")
    
    # flush نهایی
    logger.flush()
    
    # ذخیره مدل
    agent.save(f"{logger.session_dir}/agent_final.pth")
    
    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")
    print(f"📁 Logs saved to: {logger.session_dir}")
    print("=" * 60)
    
    # نمایش خلاصه
    latest_metrics = logger.get_latest_metrics(n=10)
    print("\n📊 Summary:")
    summary = latest_metrics.get('summary', {})
    print(f"   - Total episodes: {summary.get('total_episodes', 0)}")
    print(f"   - Avg reward (last 10): {summary.get('avg_reward_last_100', 0):.2f}")
    print(f"   - Best reward: {summary.get('best_reward', 0):.2f}")


if __name__ == "__main__":
    test_agent_with_logging()
