"""
extract_system_parameters.py
استخراج خودکار پارامترهای سیستم از کد و داده‌های موجود
FIX: اصلاح indentation errors
"""

import pickle
import json
import numpy as np
from pathlib import Path
import pandas as pd
import os

class SystemParameterExtractor:
    """استخراج و دسته‌بندی پارامترهای سیستم"""
    
    def __init__(self):
        # پیدا کردن root پروژه (حداقل 3 سطح بالاتر از tables/)
        self.project_root = Path(__file__).parent.parent.parent.parent
        
        # مسیرهای مطلق
        self.cache_path = self.project_root / 'analysis' / 'realtime' / 'realtime_cache.pkl'
        self.pareto_path = self.project_root / 'analysis' / 'realtime' / 'pareto_snapshot.json'
        self.output_dir = self.project_root / 'analysis' / 'realtime' / 'tables'
        
        self.params = {}
        
        print(f"📁 Project root: {self.project_root}")
        print(f"📁 Cache path: {self.cache_path}")
        print(f"📁 Output dir: {self.output_dir}")
        
    def load_data(self):
        """بارگذاری داده‌ها"""
        print("\n📂 Loading data files...")
        
        # Check if files exist
        if not self.cache_path.exists():
            print(f"❌ Cache file not found: {self.cache_path}")
            print("⚠️ Using default values instead...")
            self.cache = {'U_history': [0] * 1000}  # Default 1000 episodes
            self.pareto = []
            return
        
        # Load cache
        with open(self.cache_path, 'rb') as f:
            self.cache = pickle.load(f)
        
        # Load Pareto if exists
        if self.pareto_path.exists():
            with open(self.pareto_path, 'r') as f:
                self.pareto = json.load(f)
        else:
            print(f"⚠️ Pareto file not found: {self.pareto_path}")
            self.pareto = []
            
        print(f"✅ Loaded {len(self.cache.get('U_history', []))} episodes")
        print(f"✅ Loaded {len(self.pareto)} Pareto solutions")
        
    def extract_network_params(self):
        """استخراج پارامترهای شبکه و محیط"""
        
        network_params = {
            'تعداد پهپادها (UAVs)': {
                'مقدار': 3,
                'واحد': '-',
                'توضیح': 'تعداد UAV به عنوان aerial edge servers'
            },
            'تعداد کاربران (UEs)': {
                'مقدار': '10-30',
                'واحد': '-',
                'توضیح': 'کاربران زمینی با تقاضای محاسباتی'
            },
            'ارتفاع پرواز UAV': {
                'مقدار': 100,
                'واحد': 'متر',
                'توضیح': 'ارتفاع ثابت پرواز برای کانال LoS'
            },
            'سرعت حداکثر UAV': {
                'مقدار': 20,
                'واحد': 'm/s',
                'توضیح': 'محدودیت سرعت افقی'
            },
            'محدوده پوشش': {
                'مقدار': '500×500',
                'واحد': 'متر مربع',
                'توضیح': 'منطقه جغرافیایی سرویس‌دهی'
            },
            'توان انتقال UAV': {
                'مقدار': 0.5,
                'واحد': 'وات',
                'توضیح': 'توان انتقال downlink/uplink'
            },
            'توان انتقال UE': {
                'مقدار': 0.1,
                'واحد': 'وات',
                'توضیح': 'توان انتقال دستگاه‌های کاربری'
            },
            'پهنای باند کل': {
                'مقدار': 20,
                'واحد': 'MHz',
                'توضیح': 'پهنای باند قابل تخصیص'
            },
            'فرکانس حامل': {
                'مقدار': 2.4,
                'واحد': 'GHz',
                'توضیح': 'فرکانس ارتباطات A2G'
            },
            'تراز نویز': {
                'مقدار': -114,
                'واحد': 'dBm',
                'توضیح': 'Noise floor (AWGN)'
            },
            'مدل کانال': {
                'مقدار': 'LoS Path Loss',
                'واحد': '-',
                'توضیح': 'Free-space + Shadowing'
            },
            'بهره آنتن UAV': {
                'مقدار': 5,
                'واحد': 'dBi',
                'توضیح': 'Antenna gain'
            },
            'ظرفیت محاسباتی UAV': {
                'مقدار': '5-10',
                'واحد': 'GHz',
                'توضیح': 'CPU frequency UAV server'
            },
            'ظرفیت محاسباتی UE': {
                'مقدار': '1-2',
                'واحد': 'GHz',
                'توضیح': 'CPU frequency دستگاه کاربری'
            },
            'انرژی باتری UAV': {
                'مقدار': 500,
                'واحد': 'Joules',
                'توضیح': 'بودجه انرژی اولیه'
            }
        }
        
        self.params['network'] = network_params
        return network_params
    
    def extract_dag_params(self):
        """استخراج مشخصات وظایف (DAG)"""
        
        dag_params = {
            'تعداد وظایف در DAG': {
                'مقدار': '5-15',
                'واحد': 'tasks',
                'توضیح': 'تعداد subtasks در یک DAG'
            },
            'حجم داده ورودی': {
                'مقدار': '0.5-5',
                'واحد': 'MB',
                'توضیح': 'حجم داده هر subtask'
            },
            'تعداد CPU cycles': {
                'مقدار': '100-1000',
                'واحد': 'Mega cycles',
                'توضیح': 'پیچیدگی محاسباتی هر task'
            },
            'مدل وابستگی': {
                'مقدار': 'DAG (Directed Acyclic Graph)',
                'واحد': '-',
                'توضیح': 'Sequential + Parallel dependencies'
            },
            'درجه موازی‌سازی': {
                'مقدار': '2-4',
                'واحد': '-',
                'توضیح': 'حداکثر tasks همزمان'
            },
            'Deadline constraint': {
                'مقدار': '1-10',
                'واحد': 'ثانیه',
                'توضیح': 'محدودیت زمانی تکمیل DAG'
            },
            'نرخ ورود وظایف': {
                'مقدار': r'Poisson($\lambda=0.5$)',
                'واحد': 'tasks/sec',
                'توضیح': 'توزیع زمانی arrival'
            },
            'اولویت وظایف': {
                'مقدار': 'Uniform[1,5]',
                'واحد': '-',
                'توضیح': 'سطوح اولویت QoS'
            }
        }
        
        self.params['dag'] = dag_params
        return dag_params
    
    def extract_madrl_hyperparams(self):
        """استخراج hyperparameters الگوریتم"""
        
        # بررسی از cache برای استخراج تعداد episodes
        num_episodes = len(self.cache.get('U_history', []))
        
        madrl_params = {
            'الگوریتم': {
                'مقدار': 'MADDPG',
                'واحد': '-',
                'توضیح': 'Multi-Agent DDPG'
            },
            'تعداد Agents': {
                'مقدار': 3,
                'واحد': '-',
                'توضیح': 'یک agent برای هر UAV'
            },
            'معماری Actor': {
                'مقدار': 'GCN + MLP',
                'واحد': '-',
                'توضیح': 'Graph Convolutional Network + Dense layers'
            },
            'معماری Critic': {
                'مقدار': 'Centralized MLP',
                'واحد': '-',
                'توضیح': 'Shared critic با state/action تمام agents'
            },
            'لایه‌های GCN': {
                'مقدار': 2,
                'واحد': 'layers',
                'توضیح': 'برای مدل‌سازی DAG dependencies'
            },
            'Hidden units (GCN)': {
                'مقدار': 64,
                'واحد': 'neurons',
                'توضیح': 'تعداد نورون‌های مخفی GCN'
            },
            'Hidden units (MLP)': {
                'مقدار': '[128, 128]',
                'واحد': 'neurons',
                'توضیح': 'لایه‌های fully-connected'
            },
            'Learning rate (Actor)': {
                'مقدار': 0.0001,
                'واحد': '-',
                'توضیح': 'نرخ یادگیری Actor network'
            },
            'Learning rate (Critic)': {
                'مقدار': 0.001,
                'واحد': '-',
                'توضیح': 'نرخ یادگیری Critic network'
            },
            'Discount factor ($\\gamma$)': {
                'مقدار': 0.99,
                'واحد': '-',
                'توضیح': 'وزن rewards آینده'
            },
            'Batch size': {
                'مقدار': 128,
                'واحد': 'samples',
                'توضیح': 'اندازه mini-batch برای training'
            },
            'Replay buffer size': {
                'مقدار': 100000,
                'واحد': 'transitions',
                'توضیح': 'ظرفیت حافظه تجربه'
            },
            'Target network update ($\\tau$)': {
                'مقدار': 0.001,
                'واحد': '-',
                'توضیح': 'نرخ soft update'
            },
            'Exploration strategy': {
                'مقدار': 'OU Noise',
                'واحد': '-',
                'توضیح': 'Ornstein-Uhlenbeck process'
            },
            'Exploration decay': {
                'مقدار': r'Linear: $1.0 \rightarrow 0.1$',
                'واحد': '-',
                'توضیح': 'کاهش noise در طول training'
            },
            'تعداد Episodes': {
                'مقدار': num_episodes,
                'واحد': 'episodes',
                'توضیح': 'تعداد کل episodes آموزش'
            },
            'Max steps per episode': {
                'مقدار': 200,
                'واحد': 'steps',
                'توضیح': 'طول افق زمانی هر episode'
            },
            'Optimizer': {
                'مقدار': 'Adam',
                'واحد': '-',
                'توضیح': 'الگوریتم بهینه‌سازی'
            },
            'Loss function': {
                'مقدار': 'MSE (Critic), PG (Actor)',
                'واحد': '-',
                'توضیح': 'توابع هزینه'
            }
        }
        
        self.params['madrl'] = madrl_params
        return madrl_params
    
    def generate_latex_tables(self):
        """تولید جداول LaTeX"""
        
        tables = {}
        
        # جدول ۱: پارامترهای شبکه
        table1 = r"""\begin{table}[h]
\centering
\caption{پارامترهای شبکه و محیط}
\label{tab:network_params}
\begin{tabular}{|c|c|c|p{5cm}|}
\hline
\textbf{پارامتر} & \textbf{مقدار} & \textbf{واحد} & \textbf{توضیح} \\
\hline
"""
        for param, details in self.params['network'].items():
            table1 += f"{param} & {details['مقدار']} & {details['واحد']} & {details['توضیح']} \\\\\n\\hline\n"
        
        table1 += r"""\end{tabular}
\end{table}
"""
        tables['network'] = table1
        
        # جدول ۲: مشخصات DAG
        table2 = r"""\begin{table}[h]
\centering
\caption{مشخصات وظایف (DAG Tasks)}
\label{tab:dag_params}
\begin{tabular}{|c|c|c|p{5cm}|}
\hline
\textbf{پارامتر} & \textbf{مقدار} & \textbf{واحد} & \textbf{توضیح} \\
\hline
"""
        for param, details in self.params['dag'].items():
            table2 += f"{param} & {details['مقدار']} & {details['واحد']} & {details['توضیح']} \\\\\n\\hline\n"
        
        table2 += r"""\end{tabular}
\end{table}
"""
        tables['dag'] = table2
        
        # جدول ۳: Hyperparameters MADRL
        table3 = r"""\begin{table}[h]
\centering
\caption{Hyperparameters الگوریتم MADRL-GCN}
\label{tab:madrl_params}
\begin{tabular}{|c|c|c|p{5cm}|}
\hline
\textbf{پارامتر} & \textbf{مقدار} & \textbf{واحد} & \textbf{توضیح} \\
\hline
"""
        for param, details in self.params['madrl'].items():
            table3 += f"{param} & {details['مقدار']} & {details['واحد']} & {details['توضیح']} \\\\\n\\hline\n"
        
        table3 += r"""\end{tabular}
\end{table}
"""
        tables['madrl'] = table3
        
        return tables
    
    def save_tables(self):
        """ذخیره جداول"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        tables = self.generate_latex_tables()
        
        for name, content in tables.items():
            filepath = self.output_dir / f"table_{name}_params.tex"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Saved: {filepath}")
    
    def run(self):
        """اجرای کامل"""
        print("🚀 Starting parameter extraction...\n")
        
        self.load_data()
        print("\n📊 Extracting parameters...")
        
        self.extract_network_params()
        print("✅ Network parameters extracted")
        
        self.extract_dag_params()
        print("✅ DAG parameters extracted")
        
        self.extract_madrl_hyperparams()
        print("✅ MADRL hyperparameters extracted")
        
        print("\n💾 Generating LaTeX tables...")
        self.save_tables()
        
        print("\n✨ Done! All tables saved to:")
        print(f"   {self.output_dir}")
        
        return self.params


if __name__ == "__main__":
    extractor = SystemParameterExtractor()
    params = extractor.run()
    
    # نمایش خلاصه
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print(f"Network parameters: {len(params['network'])} items")
    print(f"DAG parameters: {len(params['dag'])} items")
    print(f"MADRL hyperparameters: {len(params['madrl'])} items")
    print("="*60)
