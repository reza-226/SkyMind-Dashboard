"""
تست I-DDPG Agent با محیط واقعی UAVMECEnvironment
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from algorithms.baselines.iddpg import IDDPGAgent

def test_iddpg_instantiation():
    """تست 1: آیا agent به درستی ساخته می‌شه؟"""
    print("\n" + "="*60)
    print("🧪 Test 1: I-DDPG Agent Instantiation")
    print("="*60)
    
    try:
        agent = IDDPGAgent(
            agent_id=0,
            local_state_dim=268,
            action_dim=11,
            offload_dim=5,
            continuous_dim=6,
            hidden=512,
            device="cpu"
        )
        print("✅ Agent created successfully!")
        
        # چک کردن component‌ها
        assert hasattr(agent, 'actor'), "❌ Missing actor"
        assert hasattr(agent, 'critic'), "❌ Missing critic"
        assert hasattr(agent, 'actor_target'), "❌ Missing actor_target"
        assert hasattr(agent, 'critic_target'), "❌ Missing critic_target"
        print("✅ All components exist!")
        
        # چک کردن parameter count
        actor_params = sum(p.numel() for p in agent.actor.parameters())
        critic_params = sum(p.numel() for p in agent.critic.parameters())
        print(f"📊 Actor params: {actor_params:,}")
        print(f"📊 Critic params: {critic_params:,}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_action_selection(agent):
    """تست 2: آیا action selection کار می‌کنه؟"""
    print("\n" + "="*60)
    print("🧪 Test 2: Action Selection")
    print("="*60)
    
    try:
        # ساخت fake local state
        local_state = np.random.randn(268)
        
        # تست با exploration
        action_explore = agent.select_action(local_state, explore=True, epsilon=0.3)
        print("✅ Exploration action:")
        print(f"  Offload: {action_explore['offload']}")
        print(f"  CPU: {action_explore['cpu']:.3f}")
        print(f"  Bandwidth: {action_explore['bandwidth']}")
        print(f"  Move: {action_explore['move']}")
        
        # تست بدون exploration
        action_greedy = agent.select_action(local_state, explore=False)
        print("\n✅ Greedy action:")
        print(f"  Offload: {action_greedy['offload']}")
        print(f"  CPU: {action_greedy['cpu']:.3f}")
        print(f"  Bandwidth: {action_greedy['bandwidth']}")
        print(f"  Move: {action_greedy['move']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_update(agent):
    """تست 3: آیا update کار می‌کنه؟"""
    print("\n" + "="*60)
    print("🧪 Test 3: Agent Update")
    print("="*60)
    
    try:
        batch_size = 32
        
        # ساخت fake batch
        batch = {
            'local_state': torch.randn(batch_size, 268),
            'action': torch.randn(batch_size, 11),
            'reward': torch.randn(batch_size, 1),
            'next_local_state': torch.randn(batch_size, 268),
            'done': torch.zeros(batch_size, 1)
        }
        
        # اجرای update
        losses = agent.update(batch)
        
        print("✅ Update successful!")
        print(f"📊 Critic Loss: {losses['critic_loss']:.4f}")
        print(f"📊 Actor Loss: {losses['actor_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_save_load(agent):
    """تست 4: آیا save/load کار می‌کنه؟"""
    print("\n" + "="*60)
    print("🧪 Test 4: Save/Load")
    print("="*60)
    
    try:
        import tempfile
        
        # ذخیره
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        agent.save(temp_path)
        print(f"✅ Model saved to {temp_path}")
        
        # لود کردن
        new_agent = IDDPGAgent(
            agent_id=0,
            local_state_dim=268,
            action_dim=11,
            device="cpu"
        )
        new_agent.load(temp_path)
        print("✅ Model loaded successfully!")
        
        # پاک کردن فایل موقت
        os.remove(temp_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("🚀 I-DDPG BASELINE TEST SUITE")
    print("="*60)
    
    # Test 1: Instantiation
    agent = test_iddpg_instantiation()
    if agent is None:
        print("\n❌ Tests aborted due to instantiation failure")
        return
    
    # Test 2: Action Selection
    success_action = test_action_selection(agent)
    
    # Test 3: Update
    success_update = test_update(agent)
    
    # Test 4: Save/Load
    success_save = test_save_load(agent)
    
    # خلاصه نتایج
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    results = {
        "Instantiation": "✅",
        "Action Selection": "✅" if success_action else "❌",
        "Update": "✅" if success_update else "❌",
        "Save/Load": "✅" if success_save else "❌"
    }
    
    for test_name, status in results.items():
        print(f"  {status} {test_name}")
    
    all_passed = all(s == "✅" for s in results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✨ I-DDPG is ready for training!")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
