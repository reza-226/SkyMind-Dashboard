"""
رابط خط فرمان (CLI) برای Analysis Toolkit
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import sys

from .analyzers.training_analyzer import TrainingAnalyzer
from .analyzers.model_evaluator import ModelEvaluator
from .analyzers.action_analyzer import ActionAnalyzer
from .analyzers.comparison import ComparisonAnalyzer
from .reporters.html_reporter import HTMLReporter
from .reporters.markdown_reporter import MarkdownReporter


class AnalysisCLI:
    """مدیریت رابط خط فرمان برای تحلیل"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.results = {}
    
    def _create_parser(self):
        """ایجاد parser برای آرگومان‌های خط فرمان"""
        parser = argparse.ArgumentParser(
            description='UAV-MEC Training Analysis Toolkit',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
مثال‌های استفاده:
  
  # تحلیل کامل
  python -m analysis_toolkit --model results/experiment_1/best_model.pth --full-analysis
  
  # فقط ارزیابی مدل
  python -m analysis_toolkit --model results/experiment_1/best_model.pth --evaluate --episodes 50
  
  # مقایسه با استراتژی تصادفی
  python -m analysis_toolkit --model results/experiment_1/best_model.pth --compare-random --episodes 30
  
  # تولید گزارش HTML
  python -m analysis_toolkit --model results/experiment_1/best_model.pth --full-analysis --html
            """
        )
        
        # آرگومان‌های اصلی
        parser.add_argument('--model', type=str, required=True,
                          help='مسیر فایل مدل آموزش‌دیده (best_model.pth)')
        
        parser.add_argument('--full-analysis', action='store_true',
                          help='اجرای تحلیل کامل (همه بخش‌ها)')
        
        # آرگومان‌های تحلیل
        parser.add_argument('--evaluate', action='store_true',
                          help='ارزیابی مدل در محیط')
        
        parser.add_argument('--compare-random', action='store_true',
                          help='مقایسه با استراتژی تصادفی')
        
        parser.add_argument('--analyze-training', action='store_true',
                          help='تحلیل تاریخچه آموزش')
        
        parser.add_argument('--analyze-actions', action='store_true',
                          help='تحلیل توزیع اکشن‌ها')
        
        # تنظیمات
        parser.add_argument('--episodes', type=int, default=50,
                          help='تعداد اپیزودها برای ارزیابی (پیش‌فرض: 50)')
        
        parser.add_argument('--detailed', action='store_true',
                          help='نمایش جزئیات کامل هر اپیزود')
        
        # خروجی
        parser.add_argument('--output-dir', type=str, default='analysis_results',
                          help='پوشه خروجی برای نتایج')
        
        parser.add_argument('--html', action='store_true',
                          help='تولید گزارش HTML')
        
        parser.add_argument('--markdown', action='store_true',
                          help='تولید گزارش Markdown')
        
        return parser
    
    def run(self, args=None):
        """اجرای تحلیل بر اساس آرگومان‌ها"""
        args = self.parser.parse_args(args)
        
        # بررسی وجود مدل
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"❌ Error: Model file not found: {model_path}")
            sys.exit(1)
        
        # ایجاد پوشه خروجی
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"🚀 UAV-MEC Analysis Toolkit")
        print(f"{'='*70}\n")
        print(f"📂 Model: {model_path}")
        print(f"📊 Output: {output_dir}\n")
        
        # فعال‌سازی همه بخش‌ها در حالت full-analysis
        if args.full_analysis:
            args.evaluate = True
            args.compare_random = True
            args.analyze_training = True
            args.analyze_actions = True
        
        # 1. ارزیابی مدل
        if args.evaluate:
            print("🔍 Step 1/4: Evaluating model...")
            evaluator = ModelEvaluator(model_path)
            eval_results = evaluator.evaluate(
                num_episodes=args.episodes,
                detailed=args.detailed
            )
            self.results['evaluation'] = eval_results
            self._print_evaluation_summary(eval_results)
        
        # 2. مقایسه با تصادفی
        if args.compare_random:
            print("\n📊 Step 2/4: Comparing with random strategy...")
            comparator = ComparisonAnalyzer(model_path)
            comparison_results = comparator.compare(
                num_episodes=args.episodes
            )
            self.results['comparison'] = comparison_results
            self._print_comparison_summary(comparison_results)
        
        # 3. تحلیل تاریخچه آموزش
        if args.analyze_training:
            print("\n📈 Step 3/4: Analyzing training history...")
            training_analyzer = TrainingAnalyzer(model_path.parent)
            training_results = training_analyzer.analyze()
            self.results['training'] = training_results
            self._print_training_summary(training_results)
        
        # 4. تحلیل اکشن‌ها
        if args.analyze_actions:
            print("\n🎯 Step 4/4: Analyzing action distributions...")
            action_analyzer = ActionAnalyzer(model_path)
            action_results = action_analyzer.analyze(num_episodes=args.episodes)
            self.results['actions'] = action_results
            self._print_action_summary(action_results)
        
        # ذخیره نتایج
        results_file = output_dir / 'analysis_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Results saved to: {results_file}")
        
        # تولید گزارش‌ها
        if args.html:
            print("\n📝 Generating HTML report...")
            html_reporter = HTMLReporter(output_dir)
            html_file = html_reporter.generate(self.results)
            print(f"✅ HTML report: {html_file}")
        
        if args.markdown:
            print("\n📝 Generating Markdown report...")
            md_reporter = MarkdownReporter(output_dir)
            md_file = md_reporter.generate(self.results)
            print(f"✅ Markdown report: {md_file}")
        
        print(f"\n{'='*70}")
        print(f"✅ Analysis completed successfully!")
        print(f"{'='*70}\n")
    
    def _print_evaluation_summary(self, results):
        """چاپ خلاصه ارزیابی"""
        stats = results['statistics']
        print(f"\n  📊 Evaluation Results:")
        print(f"     Episodes: {results['num_episodes']}")
        print(f"     Mean Reward: {stats['mean_reward']:.2f}")
        print(f"     Std Reward: {stats['std_reward']:.2f}")
        print(f"     Min Reward: {stats['min_reward']:.2f}")
        print(f"     Max Reward: {stats['max_reward']:.2f}")
    
    def _print_comparison_summary(self, results):
        """چاپ خلاصه مقایسه"""
        print(f"\n  📊 Comparison Results:")
        print(f"     Trained Model: {results['trained_model']['mean']:.2f} ± {results['trained_model']['std']:.2f}")
        print(f"     Random Policy: {results['random_policy']['mean']:.2f} ± {results['random_policy']['std']:.2f}")
        print(f"     Improvement: {results['improvement']:.2f}%")
    
    def _print_training_summary(self, results):
        """چاپ خلاصه آموزش"""
        if 'error' in results:
            print(f"\n  ⚠️  Training analysis not available: {results['error']}")
            return
        
        print(f"\n  📊 Training Summary:")
        print(f"     Total Episodes: {results['total_episodes']}")
        print(f"     Best Reward: {results['best_reward']:.2f}")
        print(f"     Final Reward: {results['final_reward']:.2f}")
    
    def _print_action_summary(self, results):
        """چاپ خلاصه اکشن‌ها"""
        print(f"\n  📊 Action Distribution:")
        offload_dist = results['offload_distribution']
        for location, count in offload_dist.items():
            print(f"     {location}: {count} times")


def run_analysis():
    """نقطه ورود اصلی برای اجرای تحلیل"""
    cli = AnalysisCLI()
    cli.run()


if __name__ == '__main__':
    run_analysis()
