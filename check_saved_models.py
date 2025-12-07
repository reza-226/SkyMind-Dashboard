import os
from pathlib import Path
import torch

def check_models():
    """بررسی مدل‌های ذخیره شده"""
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ پوشه models یافت نشد!")
        return
    
    print("="*80)
    print("📁 مدل‌های ذخیره شده:")
    print("="*80)
    
    for stage_dir in sorted(models_dir.iterdir()):
        if stage_dir.is_dir():
            print(f"\n🎯 {stage_dir.name}:")
            
            for checkpoint_dir in sorted(stage_dir.iterdir()):
                if checkpoint_dir.is_dir():
                    print(f"  └─ {checkpoint_dir.name}:")
                    
                    # بررسی فایل‌های داخل checkpoint
                    for model_file in sorted(checkpoint_dir.glob("*.pth")):
                        file_size = model_file.stat().st_size / 1024  # KB
                        
                        # بارگذاری و بررسی ابعاد
                        try:
                            state_dict = torch.load(
                                model_file, 
                                map_location='cpu',
                                weights_only=True
                            )
                            
                            # استخراج ابعاد از اولین لایه
                            first_layer_key = list(state_dict.keys())[0]
                            if 'weight' in first_layer_key:
                                dims = state_dict[first_layer_key].shape
                                print(f"      ├─ {model_file.name}: "
                                      f"{file_size:.1f} KB | dims={dims}")
                            else:
                                print(f"      ├─ {model_file.name}: "
                                      f"{file_size:.1f} KB")
                        
                        except Exception as e:
                            print(f"      ├─ {model_file.name}: "
                                  f"{file_size:.1f} KB | Error: {e}")
    
    print("\n" + "="*80)

def check_tensorboard_logs():
    """بررسی لاگ‌های TensorBoard"""
    
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        print("\n❌ پوشه runs یافت نشد!")
        return
    
    print("\n📊 لاگ‌های TensorBoard:")
    print("="*80)
    
    for run_dir in sorted(runs_dir.iterdir()):
        if run_dir.is_dir():
            event_files = list(run_dir.glob("events.out.tfevents.*"))
            if event_files:
                event_file = event_files[0]
                file_size = event_file.stat().st_size / 1024  # KB
                print(f"  └─ {run_dir.name}: {file_size:.1f} KB")
    
    print("="*80)
    print("\n💡 برای مشاهده گراف‌ها:")
    print("   tensorboard --logdir=runs")

if __name__ == "__main__":
    check_models()
    check_tensorboard_logs()
