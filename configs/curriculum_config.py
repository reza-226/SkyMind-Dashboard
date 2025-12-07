"""
configs/curriculum_config.py
پیکربندی Curriculum Learning
"""

# Training Configuration
TRAINING_CONFIG = {
    # Device
    'device': 'cpu',  # تغییر به 'cuda' اگر GPU داری
    
    # Hyperparameters
    'lr_actor': 1e-4,
    'lr_critic': 5e-4,          # کاهش از 1e-3
    'gamma': 0.99,               # افزایش از 0.95
    'tau': 0.005,                # کاهش از 0.01
    
    # Buffer
    'buffer_size': 1_000_000,    # افزایش از 100K
    'batch_size': 256,           # افزایش از 64
    'min_buffer_size': 1000,
    
    # Exploration
    'exploration_noise': 0.1,
    'noise_decay': 0.9995,
    'min_noise': 0.01,
    
    # Logging
    'log_frequency': 100,
    'eval_frequency': 500,
    'save_frequency': 1000,
    
    # Network Architecture
    'hidden_dim': 256,           # افزایش از 128
}

# Curriculum Stages
CURRICULUM_STAGES = [
    {
        'name': 'Level1',
        'episodes': 5000,
        'env_config': {
            # ❌ 'env_name': 'simple_tag_v3',  # حذف شده!
            'num_good': 1,
            'num_adversaries': 1,
            'num_obstacles': 0,
        },
        'description': 'Basic: 1 Agent vs 1 Adversary',
    },
    {
        'name': 'Level2',
        'episodes': 7000,
        'env_config': {
            # ❌ 'env_name': 'simple_tag_v3',  # حذف شده!
            'num_good': 2,
            'num_adversaries': 1,
            'num_obstacles': 0,
        },
        'description': 'Medium: 2 Agents vs 1 Adversary',
    },
    {
        'name': 'Level3',
        'episodes': 10000,
        'env_config': {
            # ❌ 'env_name': 'simple_tag_v3',  # حذف شده!
            'num_good': 2,
            'num_adversaries': 1,
            'num_obstacles': 2,
        },
        'description': 'Hard: 2 Agents vs 1 Adversary + Obstacles',
    },
]
