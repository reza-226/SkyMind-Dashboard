"""
Training Script for I-DDPG Baseline
Compatible with Real UAV-MEC Environment
✅ با Logging کامل پارامترهای 4 لایه (Ground/Edge/Fog/Cloud)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
from collections import deque

from algorithms.baselines.iddpg import IDDPGAgent

def extract_layer_metrics(state):
    """
    استخراج پارامترهای 4 لایه از State
    
    State structure (35 dims):
    - [0:6]: UAV positions (x1,y1,z1, x2,y2,z2)
    - [6:8]: Battery levels (2 UAVs)
    - [8:18]: Task info (10 tasks)
    - [18:22]: Communication delays [Ground, Edge, Fog, Cloud]
    - [22:26]: Energy consumption [Ground, Edge, Fog, Cloud]
    - [26:30]: User distances [Ground, Edge, Fog, Cloud]
    - [30:35]: QoS metrics (latency, throughput, reliability, packet_loss, jitter)
    """
    
    if len(state) < 35:
        # اگر state کوتاه‌تر باشد، مقادیر پیش‌فرض
        return {
            'communication_delays': {'ground': 0, 'edge': 0, 'fog': 0, 'cloud': 0},
            'energy_consumption': {'ground': 0, 'edge': 0, 'fog': 0, 'cloud': 0},
            'user_distances': {'ground': 0, 'edge': 0, 'fog': 0, 'cloud': 0},
            'qos_metrics': {'latency': 0, 'throughput': 0, 'reliability': 0, 'packet_loss': 0, 'jitter': 0}
        }
    
    metrics = {
        'uav_positions': {
            'uav1': {'x': float(state[0]), 'y': float(state[1]), 'z': float(state[2])},
            'uav2': {'x': float(state[3]), 'y': float(state[4]), 'z': float(state[5])}
        },
        'battery_levels': {
            'uav1': float(state[6]),
            'uav2': float(state[7])
        },
        'communication_delays': {
            'ground': float(state[18]),
            'edge': float(state[19]),
            'fog': float(state[20]),
            'cloud': float(state[21])
        },
        'energy_consumption': {
            'ground': float(state[22]),
            'edge': float(state[23]),
            'fog': float(state[24]),
            'cloud': float(state[25])
        },
        'user_distances': {
            'ground': float(state[26]),
            'edge': float(state[27]),
            'fog': float(state[28]),
            'cloud': float(state[29])
        },
        'qos_metrics': {
            'latency': float(state[30]),
            'throughput': float(state[31]),
            'reliability': float(state[32]),
            'packet_loss': float(state[33]),
            'jitter': float(state[34])
        }
    }
    
    return metrics

def create_fake_env():
    """محیط ساده برای تست سریع"""
    class FakeEnv:
        def __init__(self, n_agents=3):
            self.n_agents = n_agents
            self.step_count = 0
            self.max_steps = 50
            
        def reset(self):
            self.step_count = 0
            return {i: np.random.randn(268) for i in range(self.n_agents)}
        
        def step(self, actions):
            self.step_count += 1
            rewards = {i: np.random.uniform(-50, 150) for i in range(self.n_agents)}
            next_states = {i: np.random.randn(268) for i in range(self.n_agents)}
            dones = {i: self.step_count >= self.max_steps for i in range(self.n_agents)}
            return next_states, rewards, dones, {}
    
    return FakeEnv()

def create_real_env(n_agents=3):
    """محیط واقعی UAV-MEC با استفاده از UAVMECEnvironment"""
    try:
        from environments.uav_mec_env import UAVMECEnvironment, UAVConfig, TaskConfig
        
        print("🌍 Loading UAVMECEnvironment...")
        
        # پیکربندی UAV
        uav_config = UAVConfig(
            num_uavs=n_agents,
            max_height=100.0,
            min_height=20.0,
            max_speed=15.0,
            battery_capacity=10000.0,
            computation_capacity=5.0,
            bandwidth=20.0
        )
        
        # پیکربندی Tasks
        task_config = TaskConfig(
            num_tasks=10,
            min_data_size=0.5,
            max_data_size=5.0,
            min_cpu_cycles=500.0,
            max_cpu_cycles=2000.0,
            max_delay=2.0
        )
        
        # ساخت محیط
        env = UAVMECEnvironment(
            uav_config=uav_config,
            task_config=task_config,
            grid_size=(500.0, 500.0),
            num_obstacles=5,
            difficulty='easy'
        )
        
        # ✅ Wrapper برای سازگاری با I-DDPG
        class UAVMECWrapper:
            def __init__(self, gym_env, n_agents):
                self.gym_env = gym_env
                self.n_agents = n_agents
                
            def reset(self):
                obs, _ = self.gym_env.reset()
                # تقسیم observation به n_agents بخش (هر agent: 35 dim)
                obs_per_agent = 35
                return {i: obs[i*obs_per_agent:(i+1)*obs_per_agent] 
                        for i in range(self.n_agents)}
            
            def step(self, actions):
                # تبدیل dict actions به flat array
                action_array = []
                for i in range(self.n_agents):
                    if i in actions:
                        act = actions[i]
                        if isinstance(act, dict):
                            # ✅ استخراج فقط 4 عنصر: [dx, dy, dz, offload]
                            move = act.get('move', [0.0, 0.0])
                            dz = act.get('dz', 0.0)
                            offload = 1.0 if act.get('offload', 0) > 0 else 0.0
                            
                            action_array.extend([
                                float(move[0]),
                                float(move[1]),
                                float(dz),
                                float(offload)
                            ])
                        else:
                            # اگر numpy array بود
                            action_array.extend(act[:4] if len(act) >= 4 else [0,0,0,0])
                
                action_array = np.array(action_array, dtype=np.float32)
                
                obs, reward, terminated, truncated, info = self.gym_env.step(action_array)
                
                # تقسیم observation
                obs_per_agent = 35
                next_states = {i: obs[i*obs_per_agent:(i+1)*obs_per_agent] 
                              for i in range(self.n_agents)}
                
                rewards = {i: reward for i in range(self.n_agents)}
                dones = {i: terminated or truncated for i in range(self.n_agents)}
                
                return next_states, rewards, dones, info
        
        return UAVMECWrapper(env, n_agents)
        
    except ImportError as e:
        print(f"⚠️ Import Error: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_pettingzoo_env():
    """محیط PettingZoo برای تست"""
    try:
        from pettingzoo.mpe import simple_tag_v3
        print("🎮 Loading PettingZoo simple_tag_v3 environment...")
        
        class PZWrapper:
            def __init__(self):
                self.env = simple_tag_v3.parallel_env(
                    num_good=1,
                    num_adversaries=3,
                    num_obstacles=2,
                    max_cycles=25,
                    continuous_actions=True
                )
                self.agents = self.env.possible_agents
                self.n_agents = len(self.agents)
                
            def reset(self):
                obs, _ = self.env.reset()
                return {i: obs[agent] for i, agent in enumerate(self.agents)}
            
            def step(self, actions):
                pz_actions = {}
                for i, agent in enumerate(self.agents):
                    if i in actions:
                        action = actions[i]
                        if isinstance(action, dict):
                            pz_actions[agent] = action.get('move', np.zeros(5))
                        else:
                            pz_actions[agent] = action
                
                obs, rewards, dones, truncs, infos = self.env.step(pz_actions)
                
                next_states = {i: obs.get(agent, np.zeros(268)) for i, agent in enumerate(self.agents)}
                agent_rewards = {i: rewards.get(agent, 0.0) for i, agent in enumerate(self.agents)}
                agent_dones = {i: dones.get(agent, False) or truncs.get(agent, False) 
                              for i, agent in enumerate(self.agents)}
                
                return next_states, agent_rewards, agent_dones, {}
        
        return PZWrapper()
        
    except ImportError as e:
        print(f"⚠️ PettingZoo not found: {e}")
        return None

class ReplayBuffer:
    """Replay Buffer برای I-DDPG"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, local_state, action, reward, next_local_state, done):
        self.buffer.append((local_state, action, reward, next_local_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        local_states, actions, rewards, next_local_states, dones = zip(*batch)
        
        return {
            'local_state': torch.FloatTensor(np.array(local_states)),
            'action': torch.FloatTensor(np.array(actions)),
            'reward': torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            'next_local_state': torch.FloatTensor(np.array(next_local_states)),
            'done': torch.FloatTensor(np.array(dones)).unsqueeze(1)
        }
    
    def __len__(self):
        return len(self.buffer)

def action_to_vector(action_dict):
    """تبدیل action dict به vector"""
    if isinstance(action_dict, np.ndarray):
        return action_dict
    
    # ✅ برای UAV-MEC: [dx, dy, dz, offload]
    move = action_dict.get('move', [0.0, 0.0])
    dz = action_dict.get('dz', 0.0)
    offload = 1.0 if action_dict.get('offload', 0) > 0 else 0.0
    
    return np.array([
        float(move[0]),
        float(move[1]),
        float(dz),
        float(offload)
    ], dtype=np.float32)

def train_iddpg(
    env_type='fake',
    n_episodes=500,
    n_agents=3,
    batch_size=64,
    buffer_capacity=100000,
    warmup_steps=1000,
    update_freq=1,
    lr_actor=1e-4,
    lr_critic=1e-3,
    gamma=0.99,
    tau=0.005,
    hidden=512,
    device="cpu"
):
    """Training loop for I-DDPG با Logging کامل 4 لایه"""
    
    print("\n" + "="*70)
    print("🚀 I-DDPG TRAINING START")
    print("="*70)
    print(f"Environment: {env_type.upper()}")
    print(f"Episodes: {n_episodes}")
    print(f"Agents: {n_agents}")
    print(f"Device: {device}")
    print("="*70 + "\n")
    
    # انتخاب محیط
    env = None
    
    if env_type == 'real':
        env = create_real_env(n_agents)
    elif env_type == 'pettingzoo':
        env = create_pettingzoo_env()
    
    # Fallback به Fake
    if env is None:
        print("⚠️ Requested environment not available!")
        print("🎮 Falling back to FakeEnv...\n")
        env = create_fake_env()
        env_type = 'fake'
    
    print(f"✅ Environment loaded: {type(env).__name__}\n")
    
    # تشخیص ابعاد state
    initial_states = env.reset()
    state_dim = len(initial_states[0])
    
    # ✅ تشخیص action dimension بر اساس محیط
    if env_type == 'real':
        action_dim = 4  # [dx, dy, dz, offload]
        use_simple_action = True
        offload_dim = 5
        continuous_dim = 6
    else:
        action_dim = 11  # [offload_onehot(5) + continuous(6)]
        use_simple_action = False
        offload_dim = 5
        continuous_dim = 6
    
    print(f"📊 State dimension: {state_dim}")
    print(f"📊 Action dimension: {action_dim}")
    print(f"📊 Simple action mode: {use_simple_action}\n")
    
    # ✅ Agents با پارامتر use_simple_action
    agents = [
        IDDPGAgent(
            agent_id=i,
            local_state_dim=state_dim,
            action_dim=action_dim,
            offload_dim=offload_dim,
            continuous_dim=continuous_dim,
            hidden=hidden,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            device=device,
            use_simple_action=use_simple_action
        ) for i in range(n_agents)
    ]
    
    # Replay Buffers
    buffers = [ReplayBuffer(buffer_capacity) for _ in range(n_agents)]
    
    # ✅ Metrics برای ذخیره تاریخچه
    episode_rewards = []
    best_reward = -float('inf')
    
    # ✅ ذخیره metrics چهار لایه
    layer_metrics_history = {
        'communication_delays': {'ground': [], 'edge': [], 'fog': [], 'cloud': []},
        'energy_consumption': {'ground': [], 'edge': [], 'fog': [], 'cloud': []},
        'user_distances': {'ground': [], 'edge': [], 'fog': [], 'cloud': []},
        'qos_metrics': {'latency': [], 'throughput': [], 'reliability': [], 'packet_loss': [], 'jitter': []}
    }
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/iddpg_{env_type}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    global_step = 0
    
    for episode in range(n_episodes):
        states = env.reset()
        episode_reward = 0
        step_count = 0
        
        epsilon = max(0.05, 1.0 - episode / (n_episodes * 0.5))
        
        # ✅ استخراج metrics در شروع episode (برای محیط real)
        if env_type == 'real' and episode % 10 == 0:
            agent_state = states[0]  # state اولین agent
            current_metrics = extract_layer_metrics(agent_state)
            
            # ذخیره در history
            for layer in ['ground', 'edge', 'fog', 'cloud']:
                layer_metrics_history['communication_delays'][layer].append(
                    current_metrics['communication_delays'][layer]
                )
                layer_metrics_history['energy_consumption'][layer].append(
                    current_metrics['energy_consumption'][layer]
                )
                layer_metrics_history['user_distances'][layer].append(
                    current_metrics['user_distances'][layer]
                )
            
            for qos_key in ['latency', 'throughput', 'reliability', 'packet_loss', 'jitter']:
                layer_metrics_history['qos_metrics'][qos_key].append(
                    current_metrics['qos_metrics'][qos_key]
                )
        
        while True:
            actions = {}
            for i in range(n_agents):
                action_dict = agents[i].select_action(
                    states[i], 
                    explore=True, 
                    epsilon=epsilon
                )
                actions[i] = action_dict
            
            next_states, rewards, dones, _ = env.step(actions)
            
            # ✅ ذخیره در buffer
            for i in range(n_agents):
                action_vec = action_to_vector(actions[i])
                buffers[i].add(
                    states[i],
                    action_vec,
                    rewards[i],
                    next_states[i],
                    float(dones[i])
                )
            
            episode_reward += sum(rewards.values())
            states = next_states
            step_count += 1
            global_step += 1
            
            # ✅ Update agents
            if global_step > warmup_steps and global_step % update_freq == 0:
                for i in range(n_agents):
                    if len(buffers[i]) >= batch_size:
                        batch = buffers[i].sample(batch_size)
                        loss_info = agents[i].update(batch)
            
            if all(dones.values()):
                break
        
        avg_reward = episode_reward / n_agents
        episode_rewards.append(avg_reward)
        running_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) > 0 else avg_reward
        
        # ✅ چاپ پیشرفت با جزئیات
        if episode % 10 == 0:
            print(f"\n{'='*70}")
            print(f"📊 Episode {episode:4d}/{n_episodes}")
            print(f"{'='*70}")
            print(f"💰 Reward: {avg_reward:8.2f} | Avg(100): {running_avg:8.2f}")
            print(f"🔄 Steps: {step_count:3d} | ε: {epsilon:.3f} | Global Step: {global_step}")
            
            # ✅ چاپ metrics چهار لایه (فقط برای محیط real)
            if env_type == 'real':
                metrics = extract_layer_metrics(states[0])
                
                print(f"\n📡 Communication Delays:")
                print(f"  Ground: {metrics['communication_delays']['ground']:6.4f}s | "
                      f"Edge: {metrics['communication_delays']['edge']:6.4f}s | "
                      f"Fog: {metrics['communication_delays']['fog']:6.4f}s | "
                      f"Cloud: {metrics['communication_delays']['cloud']:6.4f}s")
                
                print(f"\n⚡ Energy Consumption:")
                print(f"  Ground: {metrics['energy_consumption']['ground']:6.2f}J | "
                      f"Edge: {metrics['energy_consumption']['edge']:6.2f}J | "
                      f"Fog: {metrics['energy_consumption']['fog']:6.2f}J | "
                      f"Cloud: {metrics['energy_consumption']['cloud']:6.2f}J")
                
                print(f"\n📏 User Distances:")
                print(f"  Ground: {metrics['user_distances']['ground']:6.2f}m | "
                      f"Edge: {metrics['user_distances']['edge']:6.2f}m | "
                      f"Fog: {metrics['user_distances']['fog']:6.2f}m | "
                      f"Cloud: {metrics['user_distances']['cloud']:6.2f}m")
                
                print(f"\n📈 QoS Metrics:")
                print(f"  Latency: {metrics['qos_metrics']['latency']:6.3f}s | "
                      f"Throughput: {metrics['qos_metrics']['throughput']:6.2f}Mbps")
                print(f"  Reliability: {metrics['qos_metrics']['reliability']:6.3f} | "
                      f"Packet Loss: {metrics['qos_metrics']['packet_loss']:6.3f}% | "
                      f"Jitter: {metrics['qos_metrics']['jitter']:6.3f}ms")
                
            print(f"{'='*70}\n")
        
        # ✅ ذخیره بهترین مدل
        if avg_reward > best_reward:
            best_reward = avg_reward
            for i, agent in enumerate(agents):
                agent.save(output_dir / f"best_agent_{i}.pt")
            if episode % 50 == 0:
                print(f"  🏆 New best! Reward: {avg_reward:.2f}\n")
    
    # ✅ ذخیره مدل نهایی
    for i, agent in enumerate(agents):
        agent.save(output_dir / f"final_agent_{i}.pt")
    
    # ✅ محاسبه آمار نهایی چهار لایه
    final_layer_analysis = {}
    if env_type == 'real' and len(layer_metrics_history['communication_delays']['ground']) > 0:
        final_layer_analysis = {
            'avg_communication_delays': {
                layer: float(np.mean(values)) if len(values) > 0 else 0.0
                for layer, values in layer_metrics_history['communication_delays'].items()
            },
            'avg_energy_consumption': {
                layer: float(np.mean(values)) if len(values) > 0 else 0.0
                for layer, values in layer_metrics_history['energy_consumption'].items()
            },
            'avg_user_distances': {
                layer: float(np.mean(values)) if len(values) > 0 else 0.0
                for layer, values in layer_metrics_history['user_distances'].items()
            },
            'avg_qos_metrics': {
                key: float(np.mean(values)) if len(values) > 0 else 0.0
                for key, values in layer_metrics_history['qos_metrics'].items()
            }
        }
    
    # ✅ ذخیره تاریخچه آموزش با metrics کامل
    history = {
        'environment': env_type,
        'episode_rewards': episode_rewards,
        'best_reward': float(best_reward),
        'final_avg': float(running_avg),
        'config': {
            'n_episodes': n_episodes,
            'n_agents': n_agents,
            'batch_size': batch_size,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'use_simple_action': use_simple_action,
            'lr_actor': lr_actor,
            'lr_critic': lr_critic,
            'gamma': gamma,
            'tau': tau,
            'hidden': hidden
        },
        'layer_metrics_history': layer_metrics_history if env_type == 'real' else {},
        'final_layer_analysis': final_layer_analysis
    }
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # ✅ چاپ خلاصه نهایی
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📊 Best Reward: {best_reward:.2f}")
    print(f"📊 Final Avg (100 eps): {running_avg:.2f}")
    print(f"💾 Models saved to: {output_dir}")
    
    if env_type == 'real' and final_layer_analysis:
        print(f"\n📊 Final Layer Analysis:")
        print(f"{'='*70}")
        print(f"Communication Delays (avg):")
        for layer, value in final_layer_analysis['avg_communication_delays'].items():
            print(f"  {layer.capitalize():6s}: {value:.4f}s")
        
        print(f"\nEnergy Consumption (avg):")
        for layer, value in final_layer_analysis['avg_energy_consumption'].items():
            print(f"  {layer.capitalize():6s}: {value:.2f}J")
        
        print(f"\nUser Distances (avg):")
        for layer, value in final_layer_analysis['avg_user_distances'].items():
            print(f"  {layer.capitalize():6s}: {value:.2f}m")
    
    print("="*70 + "\n")
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train I-DDPG Baseline')
    parser.add_argument('--env', type=str, default='fake', 
                        choices=['fake', 'real', 'pettingzoo'],
                        help='Environment type')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes')
    parser.add_argument('--agents', type=int, default=3,
                        help='Number of UAV agents')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr-actor', type=float, default=1e-4,
                        help='Actor learning rate')
    parser.add_argument('--lr-critic', type=float, default=1e-3,
                        help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient')
    parser.add_argument('--hidden', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    history = train_iddpg(
        env_type=args.env,
        n_episodes=args.episodes,
        n_agents=args.agents,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        hidden=args.hidden,
        device=args.device
    )
