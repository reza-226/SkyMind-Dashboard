# fix_all_issues.py
import re

with open('train_maddpg_ultimate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. تغییر max_episodes
content = content.replace('max_episodes=1500,  # ✅ افزایش یافت: 800 → 1500', 
                         'max_episodes=2000,  # ✅ افزایش یافت: 1500 → 2000')

# 2. اصلاح training_history
old = """    training_history = {
        'episodes': [],
        'rewards': [],
        'actor_losses': [],
        'critic_losses': [],
        'noise_std': []
    }"""

new = """    training_history = {
        'episode': [],
        'reward': [],
        'critic_loss': [],
        'noise_std': [],
        'buffer_size': []
    }"""

content = content.replace(old, new)

# 3. اصلاح append
old_append = """        training_history['episodes'].append(episode)
        training_history['rewards'].append(avg_reward)
        training_history['actor_losses'].append(np.mean(actor_losses) if actor_losses else 0)
        training_history['critic_losses'].append(np.mean(critic_losses) if critic_losses else 0)
        training_history['noise_std'].append(current_noise)"""

new_append = """        training_history['episode'].append(episode + 1)
        training_history['reward'].append(avg_reward)
        training_history['critic_loss'].append(np.mean(critic_losses) if critic_losses else 0)
        training_history['noise_std'].append(current_noise)
        training_history['buffer_size'].append(len(replay_buffer))"""

content = content.replace(old_append, new_append)

# 4. اصلاح ذخیره JSON
old_save = """    # Save training history to JSON
    history_path = os.path.join(model_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)"""

new_save = """    # Save training history to JSON with Dashboard format
    history_path = os.path.join(model_dir, 'training_history.json')
    
    # تبدیل به فرمت Dashboard (episode به عنوان key)
    formatted_history = {}
    for i in range(len(training_history['episode'])):
        ep_num = str(training_history['episode'][i])
        formatted_history[ep_num] = {
            'episode': training_history['episode'][i],
            'avg_reward': training_history['reward'][i],
            'critic_loss': training_history['critic_loss'][i],
            'noise_std': training_history['noise_std'][i],
            'buffer_size': training_history['buffer_size'][i]
        }
    
    with open(history_path, 'w') as f:
        json.dump(formatted_history, f, indent=2)
    
    logger.info(f"✓ Training history saved: {history_path}")
    logger.info(f"✓ Total episodes saved: {len(formatted_history)}")"""

content = content.replace(old_save, new_save)

with open('train_maddpg_ultimate.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ همه تغییرات اعمال شد!")
print("📝 تغییرات:")
print("  1. max_episodes: 1500 → 2000")
print("  2. training_history keys اصلاح شد")
print("  3. فرمت JSON برای Dashboard بهینه شد")
