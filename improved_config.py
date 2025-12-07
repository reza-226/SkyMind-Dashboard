# improved_config.py
"""
Improved Training Configuration for Level 1
Based on analysis of degrading performance
"""

TRAINING_CONFIG = {
    # Learning rates - کاهش برای stability
    'lr_actor': 0.0001,      # کاهش از 0.001 → overfitting جلوگیری
    'lr_critic': 0.0005,     # کاهش از 0.001 → stable updates
    
    # Buffer - افزایش برای diverse experiences
    'buffer_size': 500000,   # افزایش از 100000 → better memory
    'batch_size': 128,       # ثابت یا 256 اگر RAM کافی داری
    
    # Target network - کاهش update speed
    'tau': 0.001,            # کاهش از 0.01 → smoother learning
    
    # Exploration - کاهش decay speed
    'noise_scale': 0.3,      # initial noise
    'noise_decay': 0.9995,   # کاهش از 0.995 → longer exploration
    'noise_min': 0.05,       # minimum exploration
    
    # Training parameters
    'gamma': 0.99,           # discount factor
    'episodes': 500,         # همون تعداد قبلی
    
    # Early stopping & checkpointing
    'patience': 100,         # stop if no improvement for 100 episodes
    'save_interval': 50,     # save every 50 episodes
    'eval_interval': 10,     # evaluate every 10 episodes
}

# Curriculum Learning stages (optional)
CURRICULUM_CONFIG = {
    'use_curriculum': True,
    'stages': [
        {
            'name': 'warmup',
            'episodes': 150,
            'difficulty': 'easy',
            'num_devices': 3,      # شروع با تعداد کمتر
            'noise_scale': 0.5,    # exploration بیشتر
        },
        {
            'name': 'main',
            'episodes': 250,
            'difficulty': 'medium',
            'num_devices': 5,      # تعداد واقعی
            'noise_scale': 0.3,
        },
        {
            'name': 'fine_tune',
            'episodes': 100,
            'difficulty': 'hard',
            'num_devices': 5,
            'noise_scale': 0.1,    # exploitation بیشتر
        }
    ]
}
