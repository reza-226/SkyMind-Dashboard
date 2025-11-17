# train_sequential_levels.py
import os
import sys
import json
import logging
import yaml
from pathlib import Path
from datetime import datetime

# اضافه کردن مسیر پروژه
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import از train_maddpg_ultimate
from train_maddpg_ultimate import train_maddpg

def setup_logging(log_file):
    """راه‌اندازی logging"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path='configs/levels_config.yaml'):
    """بارگذاری فایل کانفیگ"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_results(results, output_path):
    """ذخیره نتایج به JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to {output_path}")

def train_level(level_name, level_config, common_config, prev_model_dir=None):
    """آموزش یک سطح"""
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting Training: {level_config['name']}")
    print(f"{'='*80}")
    print(f"🎯 Environment: {level_config['env_name']}")
    print(f"📊 Params: {level_config['params']}")
    
    # بررسی Transfer Learning
    load_model = None
    if prev_model_dir and level_config.get('transfer_learning', {}).get('enabled', False):
        print(f"🔄 Transfer Learning ENABLED from: {prev_model_dir}")
        load_model = prev_model_dir
    else:
        print("🆕 Training from scratch")
    
    # آماده‌سازی پارامترهای آموزش
    training_params = {
        'env_name': level_config['env_name'],
        'env_kwargs': level_config['params'],  # ✅ اینجا اصلاح شد!
        'max_episodes': level_config['training']['max_episodes'],
        'batch_size': common_config['training']['batch_size'],
        'buffer_size': common_config['training']['buffer_size'],
        'lr_actor': common_config['training']['lr_actor'],
        'lr_critic': common_config['training']['lr_critic'],
        'gamma': common_config['training']['gamma'],
        'tau': common_config['training']['tau'],
        'noise_std': common_config['training']['noise_std'],
        'noise_decay': common_config['training']['noise_decay'],
        'min_noise': common_config['training']['min_noise'],
        'save_interval': common_config['training']['save_interval'],
        'model_dir': level_config['output']['model_dir'],
        'log_file': level_config['output']['log_file'],
        'device': common_config['agents']['device']
    }
    
    # اضافه کردن pretrained model اگر موجود باشد
    if load_model:
        training_params['load_pretrained'] = load_model
    
    # آموزش
    try:
        print(f"\n▶️  Starting training with parameters:")
        for key, value in training_params.items():
            if key not in ['env_kwargs']:  # Skip large dict
                print(f"    {key}: {value}")
        
        results = train_maddpg(**training_params)
        
        print(f"\n✅ Training completed for {level_config['name']}")
        print(f"📊 Best Reward: {results['best_reward']:.2f}")
        print(f"📁 Model saved to: {results['model_dir']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ذخیره نتایج
    output_data = {
        'level_name': level_name,
        'level_display_name': level_config['name'],
        'config': {
            'env_name': level_config['env_name'],
            'params': level_config['params'],
            'training': {
                'max_episodes': level_config['training']['max_episodes'],
                **common_config['training']
            }
        },
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'transfer_learning': level_config.get('transfer_learning', {})
    }
    
    save_results(output_data, level_config['output']['results_file'])
    
    # Return مسیر best model برای transfer learning بعدی
    return results['best_model_dir']

def main():
    """اجرای متوالی Training"""
    
    print("="*80)
    print("🌐 Sequential Multi-Level MADDPG Training")
    print("="*80)
    
    # بارگذاری کانفیگ
    try:
        config = load_config()
        print("✅ Config loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    common_config = config['common']
    
    # ترتیب آموزش
    training_order = ['level1_simple', 'level2_medium', 'level3_complex']
    
    prev_model_dir = None
    successful_levels = []
    
    for i, level_name in enumerate(training_order, 1):
        print(f"\n{'='*80}")
        print(f"📍 STAGE {i}/{len(training_order)}: {level_name}")
        print(f"{'='*80}\n")
        
        if level_name not in config:
            print(f"⚠️ Level '{level_name}' not found in config. Skipping...")
            continue
        
        level_config = config[level_name]
        
        # آموزش
        trained_model_dir = train_level(
            level_name=level_name,
            level_config=level_config,
            common_config=common_config,
            prev_model_dir=prev_model_dir
        )
        
        if trained_model_dir is None:
            print(f"❌ Training failed for {level_name}. Stopping.")
            break
        
        prev_model_dir = trained_model_dir
        successful_levels.append(level_name)
        print(f"\n✅ Stage {i} completed!\n")
    
    print("="*80)
    print("🎉 Training Pipeline Completed!")
    print("="*80)
    print(f"✅ Successfully trained {len(successful_levels)}/{len(training_order)} levels:")
    for level in successful_levels:
        print(f"   - {level}")
    
    print("\n📊 To view results:")
    print("   python app.py")
    print("   Then navigate to the Multi-Environment tab\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
