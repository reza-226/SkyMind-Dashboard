# test_replay_buffer_fixed.py
"""
تست Replay Buffer با پارامترهای صحیح
"""

import torch
import numpy as np
from agents.agent_maddpg_multi import ReplayBuffer

def test_replay_buffer():
    print("="*70)
    print("🔍 تست Replay Buffer با پارامترهای صحیح")
    print("="*70)
    
    # پارامترهای محیط
    state_dim = 38
    action_dim = 4
    n_agents = 3
    
    # پارامترهای Buffer
    buffer_size = 1000
    batch_size = 128
    
    print(f"\n📋 پارامترهای محیط:")
    print(f"   state_dim: {state_dim}")
    print(f"   action_dim: {action_dim}")
    print(f"   n_agents: {n_agents}")
    
    print(f"\n📋 پارامترهای Buffer:")
    print(f"   buffer_size: {buffer_size}")
    print(f"   batch_size: {batch_size}")
    
    # ساخت buffer با پارامترهای صحیح
    print(f"\n🔧 ساخت Buffer...")
    try:
        buffer = ReplayBuffer(
            buffer_size=buffer_size,
            batch_size=batch_size
        )
        print(f"   ✓ Buffer ساخته شد")
        print(f"   Type: {type(buffer)}")
    except Exception as e:
        print(f"   ✗ خطا در ساخت Buffer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # بررسی attributes
    print(f"\n🔍 بررسی attributes:")
    attrs = dir(buffer)
    important_attrs = ['add', 'sample', 'size', '__len__', 'buffer', 'position']
    for attr in important_attrs:
        if attr in attrs:
            print(f"   ✓ {attr} موجود است")
            if hasattr(buffer, attr):
                val = getattr(buffer, attr)
                if not callable(val):
                    print(f"      value: {val}")
        else:
            print(f"   ✗ {attr} موجود نیست")
    
    # ساخت داده‌های تست
    print(f"\n📊 اضافه کردن داده‌های تست...")
    
    # تست با فرمت‌های مختلف
    test_cases = [
        {
            'name': 'Test 1: States شکل (state_dim,)',
            'states': np.random.randn(state_dim),
            'actions': np.random.randn(n_agents, action_dim),
            'rewards': np.random.randn(n_agents),
            'next_states': np.random.randn(state_dim),
            'dones': np.zeros(n_agents)
        },
        {
            'name': 'Test 2: Actions flattened شکل (n_agents * action_dim,)',
            'states': np.random.randn(state_dim),
            'actions': np.random.randn(n_agents * action_dim),
            'rewards': np.random.randn(n_agents),
            'next_states': np.random.randn(state_dim),
            'dones': np.zeros(n_agents)
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n   {test['name']}")
        print(f"      States: {test['states'].shape}")
        print(f"      Actions: {test['actions'].shape}")
        print(f"      Rewards: {test['rewards'].shape}")
        print(f"      Next_states: {test['next_states'].shape}")
        print(f"      Dones: {test['dones'].shape}")
        
        try:
            buffer.add(
                test['states'],
                test['actions'],
                test['rewards'],
                test['next_states'],
                test['dones']
            )
            print(f"      ✓ داده اضافه شد")
            if hasattr(buffer, 'size'):
                print(f"      Buffer size: {buffer.size}")
            elif hasattr(buffer, '__len__'):
                print(f"      Buffer len: {len(buffer)}")
        except Exception as e:
            print(f"      ✗ خطا: {e}")
    
    # پر کردن buffer برای sample
    print(f"\n🔄 پر کردن Buffer با {batch_size} نمونه...")
    for i in range(batch_size - 2):  # -2 چون 2 تا قبلاً اضافه کردیم
        states = np.random.randn(state_dim)
        actions = np.random.randn(n_agents, action_dim)
        rewards = np.random.randn(n_agents)
        next_states = np.random.randn(state_dim)
        dones = np.zeros(n_agents)
        buffer.add(states, actions, rewards, next_states, dones)
    
    if hasattr(buffer, 'size'):
        print(f"   Buffer size: {buffer.size}")
    elif hasattr(buffer, '__len__'):
        print(f"   Buffer len: {len(buffer)}")
    
    # نمونه‌برداری
    print(f"\n🎲 Sample از Buffer...")
    try:
        # چک کردن signature متد sample
        import inspect
        sample_sig = inspect.signature(buffer.sample)
        print(f"   Sample signature: {sample_sig}")
        
        # تلاش برای sample
        sample = buffer.sample()
        
        print(f"\n📦 محتویات Sample:")
        print(f"   Type: {type(sample)}")
        
        if isinstance(sample, (list, tuple)):
            print(f"   Length: {len(sample)}")
            for i, item in enumerate(sample):
                if isinstance(item, torch.Tensor):
                    print(f"   [{i}] Tensor shape: {item.shape}, dtype: {item.dtype}, device: {item.device}")
                elif isinstance(item, np.ndarray):
                    print(f"   [{i}] NumPy shape: {item.shape}, dtype: {item.dtype}")
                else:
                    print(f"   [{i}] Type: {type(item)}")
            
            # بررسی دقیق‌تر
            if len(sample) >= 5:
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = sample[:5]
                
                print(f"\n🔍 بررسی دقیق ابعاد:")
                print(f"   States: {states_batch.shape}")
                print(f"   Actions: {actions_batch.shape}")
                print(f"   Rewards: {rewards_batch.shape}")
                print(f"   Next_states: {next_states_batch.shape}")
                print(f"   Dones: {dones_batch.shape}")
                
                # بررسی actions
                print(f"\n⚠️ بررسی کلیدی Actions:")
                if isinstance(actions_batch, torch.Tensor):
                    print(f"   Type: Tensor")
                    print(f"   Shape: {actions_batch.shape}")
                    print(f"   Ndim: {actions_batch.ndim}")
                    
                    if actions_batch.ndim == 3:
                        print(f"   ✓ 3D tensor (batch, n_agents, action_dim)")
                        flat = actions_batch.reshape(actions_batch.shape[0], -1)
                        print(f"   Flattened: {flat.shape}")
                    elif actions_batch.ndim == 2:
                        print(f"   2D tensor (batch, ?)")
                        expected_single = action_dim
                        expected_all = n_agents * action_dim
                        actual = actions_batch.shape[1]
                        
                        if actual == expected_single:
                            print(f"   ❌ تنها {action_dim} بعد - فقط یک agent!")
                        elif actual == expected_all:
                            print(f"   ✓ {expected_all} بعد - همه agents")
                        else:
                            print(f"   ⚠️ {actual} بعد - نامشخص!")
        
        print(f"\n   ✓ Sample موفق بود")
        
    except Exception as e:
        print(f"   ✗ خطا در sample: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("✅ تست Buffer تمام شد")
    print("="*70)

if __name__ == "__main__":
    test_replay_buffer()
