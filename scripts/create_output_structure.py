"""
اسکریپت ساخت ساختار پوشه‌های خروجی
"""

import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.output_manager import create_organized_results_structure


def main():
    """ایجاد ساختار پوشه‌های نتایج"""
    
    print("\n" + "="*70)
    print("🏗️  CREATING ORGANIZED OUTPUT STRUCTURE")
    print("="*70)
    
    # ایجاد ساختار
    base_dir = "results"
    results_path = create_organized_results_structure(base_dir)
    
    # نمایش ساختار ایجاد شده
    print("\n📂 Created structure:")
    print(f"\n{base_dir}/")
    print("├── level1_easy/")
    print("├── level2_medium/")
    print("├── level3_hard/")
    print("├── level4_expert/")
    print("└── final/")
    
    print("\n" + "="*70)
    print("✅ Setup complete! You can now run training with OutputManager.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
