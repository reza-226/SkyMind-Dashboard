# fix_training_history.py
import re

# خواندن فایل
with open('train_maddpg_ultimate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# اصلاح 1: تعریف دیکشنری
old_dict = """    training_history = {
        'episodes': [],
        'rewards': [],
        'actor_losses': [],
        'critic_losses': [],
        'noise_std': []
    }"""

new_dict = """    training_history = {
        'episode': [],
        'reward': [],
        'critic_loss': [],
        'noise_std': [],
        'buffer_size': []
    }"""

content = content.replace(old_dict, new_dict)

# اصلاح 2: ذخیره در حلقه
old_append = """        training_history['episodes'].append(episode)
        training_history['rewards'].append(avg_reward)
        training_history['actor_losses'].append(np.mean(actor_losses) if actor_losses else 0)
        training_history['critic_losses'].append(np.mean(critic_losses) if critic_losses else 0)
        training_history['noise_std'].append(current_noise)"""

new_append = """        training_history['episode'].append(episode + 1)
        training_history['reward'].append(avg_reward)
        training_history['critic_loss'].append(np.mean(critic_losses) if critic_losses else 0)
        training_history['noise_std'].append(current_noise)
        training_history['buffer_size'].append(len(trainer.replay_buffer))"""

content = content.replace(old_append, new_append)

# اصلاح 3: ذخیره فایل JSON
old_save = """    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)"""

new_save = """    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    
    # تبدیل به فرمت Dashboard
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
    
    logger.info(f"✓ Total episodes saved: {len(formatted_history)}")"""

content = content.replace(old_save, new_save)

# ذخیره فایل اصلاح شده
with open('train_maddpg_ultimate.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ File patched successfully!")
