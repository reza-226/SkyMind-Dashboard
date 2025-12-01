"""
تست عملکرد OutputManager
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# اضافه کردن مسیر پروژه
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.output_manager import OutputManager


def test_basic_functionality():
    """تست عملکرد اولیه"""
    
    print("\n" + "="*70)
    print("🧪 TESTING OUTPUT MANAGER")
    print("="*70 + "\n")
    
    # ایجاد OutputManager
    print("1️⃣  Creating OutputManager...")
    output_mgr = OutputManager(
        base_dir="results",
        level=1,
        difficulty="easy",
        resume=False,
        run_name="test_run"
    )
    
    # نمایش اطلاعات
    output_mgr.print_summary()
    
    # تست ذخیره config
    print("\n2️⃣  Testing config save...")
    test_config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "gamma": 0.95,
        "episodes": 5000,
    }
    output_mgr.save_config(test_config)
    
    # تست ذخیره checkpoint
    print("\n3️⃣  Testing checkpoint save...")
    dummy_checkpoint = {
        "episode": 100,
        "reward": -15.5,
        "actor_state": {"layer1.weight": np.random.randn(10, 10)},
    }
    output_mgr.save_checkpoint(dummy_checkpoint, episode=100)
    output_mgr.save_checkpoint(dummy_checkpoint, episode=200)
    
    # تست بارگذاری checkpoint
    print("\n4️⃣  Testing checkpoint load...")
    loaded = output_mgr.load_checkpoint()
    if loaded:
        print(f"  ✅ Loaded episode: {loaded['episode']}")
    
    # تست ذخیره training history
    print("\n5️⃣  Testing training history save...")
    dummy_history = pd.DataFrame({
        "episode": range(1, 11),
        "reward": np.random.randn(10) - 15,
        "actor_loss": np.random.rand(10),
    })
    output_mgr.save_training_history(dummy_history)
    
    # تست ذخیره summary
    print("\n6️⃣  Testing summary save...")
    dummy_summary = {
        "total_episodes": 10,
        "best_reward": -12.5,
        "final_reward": -14.2,
    }
    output_mgr.save_summary(dummy_summary)
    
    # تست ذخیره plot
    print("\n7️⃣  Testing plot save...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dummy_history["episode"], dummy_history["reward"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Test Reward Curve")
    output_mgr.save_plot(fig, "test_reward_curve.png")
    plt.close()
    
    # تست ذخیره best model
    print("\n8️⃣  Testing best model save...")
    dummy_model = {
        "episode": 200,
        "reward": -10.5,
        "actor_state": {"weights": np.random.randn(5, 5)},
    }
    output_mgr.save_best_model(dummy_model, level_best=False)
    
    # نمایش اطلاعات نهایی
    print("\n" + "="*70)
    output_mgr.print_summary()
    
    print("✅ All tests passed!")
    print("="*70 + "\n")
    
    return output_mgr


def test_resume_functionality():
    """تست قابلیت Resume"""
    
    print("\n" + "="*70)
    print("🔄 TESTING RESUME FUNCTIONALITY")
    print("="*70 + "\n")
    
    # Resume از run قبلی
    print("1️⃣  Attempting to resume from previous run...")
    output_mgr = OutputManager(
        base_dir="results",
        level=1,
        difficulty="easy",
        resume=True
    )
    
    output_mgr.print_summary()
    
    # بارگذاری آخرین checkpoint
    checkpoint = output_mgr.load_checkpoint()
    if checkpoint:
        print(f"  ✅ Successfully resumed from episode {checkpoint['episode']}")
    else:
        print("  ⚠️  No checkpoint found")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # تست اولیه
    mgr = test_basic_functionality()
    
    # تست Resume
    test_resume_functionality()
    
    print("\n🎉 Testing complete! Check the 'results/' folder.")
