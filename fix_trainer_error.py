# fix_trainer_final.py
import re

with open('train_maddpg_ultimate.py', 'r', encoding='utf-8') as f:
    content = f.read()

# جایگزینی trainer.replay_buffer با replay_buffer
content = content.replace(
    "training_history['buffer_size'].append(len(trainer.replay_buffer))",
    "training_history['buffer_size'].append(len(replay_buffer))"
)

with open('train_maddpg_ultimate.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ مشکل حل شد!")
print("📝 تغییر: trainer.replay_buffer → replay_buffer")
