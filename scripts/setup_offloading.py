"""
اسکریپت نصب و راه‌اندازی سیستم تخلیه محاسباتی
"""

import os
from pathlib import Path

def create_directory_structure():
    """ایجاد ساختار پوشه‌ها"""
    
    base_dir = Path(__file__).parent.parent
    
    directories = [
        "offloading_simulation",
        "experiments",
        "api",
        "results/offloading_results/simple/charts",
        "results/offloading_results/medium/charts",
        "results/offloading_results/complex/charts",
        "results/offloading_results/visualizations",
        "scripts"
    ]
    
    print("🚀 شروع ایجاد ساختار پوشه‌ها...\n")
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")
    
    print("\n✨ همه پوشه‌ها با موفقیت ایجاد شدند!\n")


def create_init_files():
    """ایجاد فایل‌های __init__.py"""
    
    base_dir = Path(__file__).parent.parent
    
    init_dirs = [
        "offloading_simulation",
        "experiments",
        "api"
    ]
    
    print("📝 ایجاد فایل‌های __init__.py...\n")
    
    for directory in init_dirs:
        init_path = base_dir / directory / "__init__.py"
        if not init_path.exists():
            init_path.write_text("# Auto-generated\n")
            print(f"✅ {directory}/__init__.py")
    
    print("\n✨ فایل‌های __init__.py آماده شدند!\n")


def create_requirements_file():
    """ایجاد فایل requirements.txt"""
    
    base_dir = Path(__file__).parent.parent
    
    requirements = """numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
pandas>=1.3.0
scipy>=1.7.0
"""
    
    req_path = base_dir / "requirements_offloading.txt"
    req_path.write_text(requirements)
    
    print("📦 فایل requirements_offloading.txt ایجاد شد!")
    print("\nبرای نصب وابستگی‌ها:")
    print("pip install -r requirements_offloading.txt\n")


def create_readme():
    """ایجاد فایل README"""
    
    base_dir = Path(__file__).parent.parent
    
    readme_content = """# Computational Offloading Simulation

## Structure

- offloading_simulation/: Core simulation modules
- experiments/: Experiment scripts
- api/: Dashboard API
- results/: Results and charts

## How to Run

1. Install dependencies:
   pip install -r requirements_offloading.txt

2. Run experiments:
   python experiments/run_offloading_experiments.py

3. View results:
   python api/offloading_api.py

## Evaluation Metrics

- Scalability: Task success rate
- Energy Efficiency: Average Joules consumed
- Latency Reduction: Comparison with local processing
- Throughput: Tasks per second

## Computational Layers

1. Ground: Local processing
2. Edge: Edge server
3. Fog: Fog computing
4. Cloud: Cloud computing
"""
    
    readme_path = base_dir / "OFFLOADING_README.md"
    readme_path.write_text(readme_content)
    
    print("📖 فایل OFFLOADING_README.md ایجاد شد!\n")


def main():
    """اجرای کامل نصب"""
    
    print("\n" + "="*60)
    print("🎯 راه‌اندازی سیستم تخلیه محاسباتی")
    print("="*60 + "\n")
    
    create_directory_structure()
    create_init_files()
    create_requirements_file()
    create_readme()
    
    print("="*60)
    print("✅ نصب با موفقیت کامل شد!")
    print("="*60)
    print("\n🚀 مراحل بعدی:")
    print("1. pip install -r requirements_offloading.txt")
    print("2. python experiments/run_offloading_experiments.py")
    print("3. مشاهده نتایج در: results/offloading_results/\n")


if __name__ == "__main__":
    main()
