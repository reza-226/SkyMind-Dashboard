# experiments/__init__.py
"""
Experiments Module
------------------
ماژول مدیریت آزمایش‌ها و ارزیابی
"""

from .scenario_loader import ScenarioLoader
from .experiment_runner import ExperimentRunner

# موقتاً ResultsAnalyzer رو کامنت می‌کنیم تا بعداً فعال کنیم
# from .results_analyzer import ResultsAnalyzer

__all__ = [
    'ScenarioLoader',
    'ExperimentRunner',
    # 'ResultsAnalyzer',  # بعداً فعال می‌شه
]

__version__ = '1.0.0'
