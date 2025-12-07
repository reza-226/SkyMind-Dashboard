# create_analysis_toolkit.py
"""
اسکریپت ایجاد ساختار کامل Analysis Toolkit
"""

import os
from pathlib import Path


def create_directory_structure():
    """ایجاد ساختار پوشه‌ها و فایل‌های خالی"""
    
    # ساختار پوشه‌ها
    structure = {
        'analysis_toolkit': {
            '__init__.py': '',
            'cli.py': '',
            'analyzers': {
                '__init__.py': '',
                'training_analyzer.py': '',
                'model_evaluator.py': '',
                'action_analyzer.py': '',
                'comparison.py': '',
            },
            'visualizers': {
                '__init__.py': '',
                'plot_training.py': '',
                'plot_actions.py': '',
                'plot_rewards.py': '',
            },
            'reporters': {
                '__init__.py': '',
                'html_reporter.py': '',
                'markdown_reporter.py': '',
            },
            'templates': {
                'report_template.html': '',
            }
        }
    }
    
    def create_structure(base_path: Path, structure: dict):
        """ایجاد بازگشتی ساختار"""
        for name, content in structure.items():
            path = base_path / name
            
            if isinstance(content, dict):
                # پوشه
                path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {path}")
                create_structure(path, content)
            else:
                # فایل
                if not path.exists():
                    path.touch()
                    print(f"📄 Created file: {path}")
                else:
                    print(f"⚠️  File already exists: {path}")
    
    # ایجاد ساختار از root پروژه
    root = Path.cwd()
    print(f"\n{'='*70}")
    print(f"🏗️  Creating Analysis Toolkit Structure")
    print(f"{'='*70}\n")
    print(f"📍 Root directory: {root}\n")
    
    create_structure(root, structure)
    
    print(f"\n{'='*70}")
    print(f"✅ Structure created successfully!")
    print(f"{'='*70}\n")
    
    # نمایش ساختار نهایی
    print("📋 Final structure:")
    print("""
analysis_toolkit/
├── __init__.py
├── cli.py
├── analyzers/
│   ├── __init__.py
│   ├── training_analyzer.py
│   ├── model_evaluator.py
│   ├── action_analyzer.py
│   └── comparison.py
├── visualizers/
│   ├── __init__.py
│   ├── plot_training.py
│   ├── plot_actions.py
│   └── plot_rewards.py
├── reporters/
│   ├── __init__.py
│   ├── html_reporter.py
│   └── markdown_reporter.py
└── templates/
    └── report_template.html
    """)
    
    print("\n🎯 Next steps:")
    print("1. Run this script to create the structure")
    print("2. I'll provide the code for each file")
    print("3. Copy-paste each code into the corresponding file")
    print("\nReady? Let's go! 🚀\n")


if __name__ == '__main__':
    create_directory_structure()
