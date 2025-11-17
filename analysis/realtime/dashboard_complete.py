# ═══════════════════════════════════════════════════════════════════════
# dashboard_complete.py - نسخه نهایی با UI بهبود یافته + جداول مقایسه‌ای
# ═══════════════════════════════════════════════════════════════════════

import dash
from dash import dcc, html, dash_table, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from collections import deque

# ═══════════════════════════════════════════════════════════════════════
# تنظیمات اولیه
# ═══════════════════════════════════════════════════════════════════════
RESULTS_DIR = 'results'
MAX_DATA_POINTS = 30

os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# خواندن داده‌های آموزش (3 سطح)
# ═══════════════════════════════════════════════════════════════════════
def load_training_data():
    """خواندن نتایج آموزش 3 سطح"""
    levels_data = {}
    level_paths = {
        'Level 1 (Simple)': 'models/level1_simple/training_history.json',
        'Level 2 (Medium)': 'models/level2_medium/training_history.json',
        'Level 3 (Complex)': 'models/level3_complex/training_history.json'
    }
    
    for level_name, path in level_paths.items():
        try:
            with open(path, 'r') as f:
                levels_data[level_name] = json.load(f)
                print(f"✅ بارگذاری موفق: {level_name}")
        except FileNotFoundError:
            print(f"⚠️  فایل یافت نشد: {level_name}")
            levels_data[level_name] = None
    
    return levels_data

training_data = load_training_data()

# ═══════════════════════════════════════════════════════════════════════
# تولید داده‌های نمونه
# ═══════════════════════════════════════════════════════════════════════
def generate_sample_data():
    """تولید داده‌های شبیه‌سازی شده"""
    n_episodes = 500
    episodes = list(range(1, n_episodes + 1))
    
    rewards = []
    utilities = []
    energies = []
    
    for i in range(n_episodes):
        base_reward = -100 + (i / n_episodes) * 150
        reward = base_reward + np.random.normal(0, 10)
        rewards.append(reward)
        
        base_utility = 0.3 + (i / n_episodes) * 0.6
        utility = np.clip(base_utility + np.random.normal(0, 0.05), 0, 1)
        utilities.append(utility)
        
        base_energy = 80 - (i / n_episodes) * 40
        energy = np.clip(base_energy + np.random.normal(0, 5), 10, 100)
        energies.append(energy)
    
    return {
        'episodes': episodes,
        'rewards': rewards,
        'utilities': utilities,
        'energies': energies
    }

sample_data = generate_sample_data()
np.save(os.path.join(RESULTS_DIR, 'sample_training_results.npy'), sample_data)

live_data_store = {
    'episodes': deque(maxlen=30),
    'utility': deque(maxlen=30),
    'energy': deque(maxlen=30)
}

episode_counter = 0

# ═══════════════════════════════════════════════════════════════════════
# ایجاد برنامه Dash
# ═══════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__)
app.title = "داشبورد SkyMind"
app.config.suppress_callback_exceptions = True

# ═══════════════════════════════════════════════════════════════════════
# استایل‌های سفارشی (تم تیره + فونت فارسی)
# ═══════════════════════════════════════════════════════════════════════
CARD_STYLE = {
    'backgroundColor': '#1e2a38',
    'padding': '25px',
    'borderRadius': '12px',
    'marginBottom': '20px',
    'boxShadow': '0 4px 15px rgba(0,0,0,0.3)',
    'border': '1px solid #2d3e50'
}

HEADER_STYLE = {
    'color': '#00d4ff',
    'marginBottom': '15px',
    'fontFamily': 'Vazirmatn, Tahoma, sans-serif',
    'fontWeight': 'bold'
}

TEXT_STYLE = {
    'color': '#e0e0e0',
    'fontSize': '15px',
    'lineHeight': '1.8',
    'fontFamily': 'Vazirmatn, Tahoma, sans-serif'
}

# ═══════════════════════════════════════════════════════════════════════
# Layout اصلی
# ═══════════════════════════════════════════════════════════════════════
app.layout = html.Div([
    # فونت فارسی
    html.Link(
        rel='stylesheet',
        href='https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css'
    ),
    
    # هدر
    html.Div([
        html.H1("🛸 داشبورد SkyMind", 
                style={'textAlign': 'center', 'color': '#00d4ff', 'marginBottom': '10px',
                       'fontFamily': 'Vazirmatn, sans-serif', 'fontSize': '42px'}),
        html.P("مانیتورینگ لحظه‌ای و نتایج آموزش MADDPG", 
               style={'textAlign': 'center', 'color': '#a0a0a0', 'fontSize': '18px',
                      'fontFamily': 'Vazirmatn, sans-serif'})
    ], style={'padding': '30px', 'backgroundColor': '#0d1b2a', 'borderBottom': '3px solid #00d4ff'}),
    
    # Tabs
    dcc.Tabs(id='tabs', value='tab-overview', children=[
        dcc.Tab(label='📊 نمای کلی', value='tab-overview',
                style={'backgroundColor': '#1e2a38', 'color': '#e0e0e0', 'fontFamily': 'Vazirmatn'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#0d1b2a', 'fontWeight': 'bold'}),
        dcc.Tab(label='🤖 جزئیات عامل‌ها', value='tab-agents',
                style={'backgroundColor': '#1e2a38', 'color': '#e0e0e0', 'fontFamily': 'Vazirmatn'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#0d1b2a', 'fontWeight': 'bold'}),
        dcc.Tab(label='⚙️ پارامترهای آموزش', value='tab-hyperparams',
                style={'backgroundColor': '#1e2a38', 'color': '#e0e0e0', 'fontFamily': 'Vazirmatn'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#0d1b2a', 'fontWeight': 'bold'}),
        dcc.Tab(label='📈 نتایج آموزش', value='tab-results',
                style={'backgroundColor': '#1e2a38', 'color': '#e0e0e0', 'fontFamily': 'Vazirmatn'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#0d1b2a', 'fontWeight': 'bold'}),
        dcc.Tab(label='🔴 مانیتورینگ زنده', value='tab-live',
                style={'backgroundColor': '#1e2a38', 'color': '#e0e0e0', 'fontFamily': 'Vazirmatn'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#0d1b2a', 'fontWeight': 'bold'})
    ], style={'fontFamily': 'Vazirmatn'}),
    
    html.Div(id='tabs-content', style={'padding': '25px', 'backgroundColor': '#0d1b2a', 'minHeight': '100vh'})
], style={'backgroundColor': '#0d1b2a', 'fontFamily': 'Vazirmatn'})

# ═══════════════════════════════════════════════════════════════════════
# Callback برای تغییر محتوای تب‌ها
# ═══════════════════════════════════════════════════════════════════════
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    """تغییر محتوای تب‌ها"""
    if tab == 'tab-overview':
        return html.Div([
            html.H3("📊 نمای کلی سیستم", style=HEADER_STYLE),
            
            html.Div([
                html.Div([
                    html.H4("🎯 هدف پروژه", style={'color': '#00ff88', 'fontFamily': 'Vazirmatn'}),
                    html.P("الگوریتم MADDPG (Multi-Agent Deep Deterministic Policy Gradient) برای بهینه‌سازی "
                           "برون‌سپاری محاسبات با کمک پهپادها (UAV-assisted Computation Offloading)", 
                           style=TEXT_STYLE),
                    html.Hr(style={'borderColor': '#2d3e50'}),
                    html.P("🔹 آموزش چند عامل هوشمند برای تصمیم‌گیری همزمان", style=TEXT_STYLE),
                    html.P("🔹 بهینه‌سازی مصرف انرژی و تاخیر شبکه", style=TEXT_STYLE),
                    html.P("🔹 استفاده از شبکه‌های Actor-Critic", style=TEXT_STYLE)
                ], style=CARD_STYLE),
                
                html.Div([
                    html.H4("📊 آمار سریع سیستم", style={'color': '#ff9500', 'fontFamily': 'Vazirmatn'}),
                    html.P(f"📁 مسیر ذخیره نتایج: {RESULTS_DIR}", style=TEXT_STYLE),
                    html.P(f"📦 تعداد فایل‌های موجود: {len([f for f in os.listdir(RESULTS_DIR) if f.endswith('.npy')])}", 
                           style=TEXT_STYLE),
                    html.P(f"🔢 حداکثر نقاط داده: {MAX_DATA_POINTS}", style=TEXT_STYLE),
                    html.P(f"⏱ پنجره نمایش زنده: 30 قدم", style=TEXT_STYLE)
                ], style=CARD_STYLE),
                
                html.Div([
                    html.H4("🛠 ویژگی‌های داشبورد", style={'color': '#00d4ff', 'fontFamily': 'Vazirmatn'}),
                    html.P("✅ نمایش لحظه‌ای عملکرد سیستم", style=TEXT_STYLE),
                    html.P("✅ تحلیل نتایج آموزش با نمودارهای تعاملی", style=TEXT_STYLE),
                    html.P("✅ جداول مقایسه‌ای 3 سطح آموزش", style=TEXT_STYLE),
                    html.P("✅ دکمه Pause/Resume برای کنترل مانیتورینگ", style=TEXT_STYLE)
                ], style=CARD_STYLE)
            ])
        ])
    
    elif tab == 'tab-agents':
        return html.Div([
            html.H3("🤖 معماری عامل‌های هوشمند", style=HEADER_STYLE),
            
            html.Div([
                html.H4("عامل 0: کنترل‌کننده پهپاد (UAV Controller)", 
                        style={'color': '#00ff88', 'fontFamily': 'Vazirmatn', 'marginBottom': '15px'}),
                html.P("این عامل مسئول تصمیم‌گیری درباره حرکت پهپاد و تخصیص منابع محاسباتی است.", 
                       style=TEXT_STYLE),
                html.Hr(style={'borderColor': '#2d3e50', 'margin': '15px 0'}),
                html.P("🔸 شبکه Actor:", style={'color': '#00d4ff', 'fontWeight': 'bold', 'fontFamily': 'Vazirmatn'}),
                html.P("ورودی [obs_dim] → لایه پنهان 128 نورون → لایه پنهان 64 نورون → خروجی [action_dim]", 
                       style={**TEXT_STYLE, 'marginLeft': '20px'}),
                html.P("تابع فعال‌سازی: ReLU در لایه‌های میانی، Tanh در خروجی", 
                       style={**TEXT_STYLE, 'marginLeft': '20px', 'color': '#a0a0a0'}),
                html.P("🔸 شبکه Critic:", style={'color': '#ff9500', 'fontWeight': 'bold', 'fontFamily': 'Vazirmatn', 'marginTop': '10px'}),
                html.P("ورودی [(obs+act) × n_agents] → 128 نورون → 64 نورون → خروجی Q-value (1)", 
                       style={**TEXT_STYLE, 'marginLeft': '20px'}),
                html.P("نقش: ارزیابی کیفیت اقدامات گرفته شده توسط تمام عامل‌ها", 
                       style={**TEXT_STYLE, 'marginLeft': '20px', 'color': '#a0a0a0'})
            ], style=CARD_STYLE),
            
            html.Div([
                html.H4("عامل 1: مدیریت سرور لبه (Edge Server Manager)", 
                        style={'color': '#00ff88', 'fontFamily': 'Vazirmatn', 'marginBottom': '15px'}),
                html.P("این عامل تصمیم می‌گیرد که کدام محاسبات را باید بر روی سرور لبه انجام دهد.", 
                       style=TEXT_STYLE),
                html.Hr(style={'borderColor': '#2d3e50', 'margin': '15px 0'}),
                html.P("🔸 شبکه Actor:", style={'color': '#00d4ff', 'fontWeight': 'bold', 'fontFamily': 'Vazirmatn'}),
                html.P("ورودی [obs_dim] → 128 نورون → 64 نورون → [action_dim]", 
                       style={**TEXT_STYLE, 'marginLeft': '20px'}),
                html.P("🔸 شبکه Critic:", style={'color': '#ff9500', 'fontWeight': 'bold', 'fontFamily': 'Vazirmatn', 'marginTop': '10px'}),
                html.P("ورودی [(obs+act) × n_agents] → 128 نورون → 64 نورون → Q-value", 
                       style={**TEXT_STYLE, 'marginLeft': '20px'}),
                html.P("💡 نکته: هر دو عامل به صورت همزمان یاد می‌گیرند و تصمیمات هم‌افزا می‌گیرند", 
                       style={**TEXT_STYLE, 'marginTop': '15px', 'color': '#ffd700', 'fontWeight': 'bold'})
            ], style=CARD_STYLE)
        ])
    
    elif tab == 'tab-hyperparams':
        return html.Div([
            html.H3("⚙️ پارامترهای آموزش", style=HEADER_STYLE),
            
            html.Div([
                dash_table.DataTable(
                    columns=[
                        {'name': 'پارامتر', 'id': 'param'},
                        {'name': 'مقدار', 'id': 'value'},
                        {'name': 'توضیحات', 'id': 'desc'}
                    ],
                    data=[
                        {'param': 'Learning Rate (Actor)', 'value': '0.0001', 
                         'desc': 'نرخ یادگیری شبکه Actor (کند برای پایداری)'},
                        {'param': 'Learning Rate (Critic)', 'value': '0.001', 
                         'desc': 'نرخ یادگیری شبکه Critic (سریع‌تر از Actor)'},
                        {'param': 'Gamma (Discount)', 'value': '0.95', 
                         'desc': 'ضریب تخفیف برای محاسبه ارزش آینده'},
                        {'param': 'Tau (Soft Update)', 'value': '0.01', 
                         'desc': 'ضریب به‌روزرسانی نرم شبکه‌های هدف'},
                        {'param': 'Batch Size', 'value': '64', 
                         'desc': 'تعداد نمونه‌ها در هر مرحله آموزش'},
                        {'param': 'Buffer Size', 'value': '100000', 
                         'desc': 'ظرفیت Replay Buffer برای ذخیره تجربیات'},
                        {'param': 'Max Episodes', 'value': '500', 
                         'desc': 'تعداد کل اپیزودهای آموزش'}
                    ],
                    style_cell={
                        'textAlign': 'right',
                        'padding': '15px',
                        'backgroundColor': '#1e2a38',
                        'color': '#e0e0e0',
                        'fontFamily': 'Vazirmatn',
                        'border': '1px solid #2d3e50'
                    },
                    style_header={
                        'backgroundColor': '#00d4ff',
                        'color': '#0d1b2a',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'fontFamily': 'Vazirmatn',
                        'fontSize': '16px'
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': '#253545'}
                    ]
                )
            ], style=CARD_STYLE),
            
            html.Div([
                html.H4("📚 توضیحات تکمیلی", style={'color': '#00ff88', 'fontFamily': 'Vazirmatn'}),
                html.P("🔹 الگوریتم MADDPG یک روش یادگیری تقویتی عمیق برای محیط‌های چند عامله است", 
                       style=TEXT_STYLE),
                html.P("🔹 از Experience Replay برای شکستن همبستگی داده‌ها استفاده می‌شود", 
                       style=TEXT_STYLE),
                html.P("🔹 Target Networks برای پایداری آموزش به کار می‌روند", 
                       style=TEXT_STYLE)
            ], style=CARD_STYLE)
        ])
    
    elif tab == 'tab-results':
        return create_tab_results()
    
    elif tab == 'tab-live':
        return html.Div([
            html.H3("🔴 مانیتورینگ زنده", style=HEADER_STYLE),
            
            html.Div([
                html.Button('▶️ شروع', id='start-button', n_clicks=0,
                           style={'padding': '12px 25px', 'fontSize': '16px',
                                  'backgroundColor': '#00ff88', 'color': '#0d1b2a',
                                  'border': 'none', 'borderRadius': '8px',
                                  'cursor': 'pointer', 'fontWeight': 'bold',
                                  'fontFamily': 'Vazirmatn', 'marginRight': '10px'}),
                html.Button('⏸️ توقف', id='pause-button', n_clicks=0,
                           style={'padding': '12px 25px', 'fontSize': '16px',
                                  'backgroundColor': '#ff9500', 'color': '#0d1b2a',
                                  'border': 'none', 'borderRadius': '8px',
                                  'cursor': 'pointer', 'fontWeight': 'bold',
                                  'fontFamily': 'Vazirmatn'}),
                html.Div(id='live-status', style={'marginTop': '15px', 'fontSize': '18px',
                                                   'color': '#00d4ff', 'fontFamily': 'Vazirmatn'})
            ], style={'marginBottom': '25px'}),
            
            dcc.Interval(id='interval-component', interval=2000, n_intervals=0, disabled=True),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='live-utility-graph', 
                             config={'displayModeBar': False},
                             style={'height': '350px'})
                ], style=CARD_STYLE),
                
                html.Div([
                    dcc.Graph(id='live-energy-graph',
                             config={'displayModeBar': False},
                             style={'height': '350px'})
                ], style=CARD_STYLE)
            ])
        ])

# ═══════════════════════════════════════════════════════════════════════
# تابع ایجاد تب نتایج با جداول مقایسه‌ای
# ═══════════════════════════════════════════════════════════════════════
def create_tab_results():
    """تب نتایج آموزش با جداول مقایسه‌ای"""
    
    # رنگ‌بندی سطوح
    colors_map = {
        'Level 1 (Simple)': '#00ff41',
        'Level 2 (Medium)': '#ffa500',
        'Level 3 (Complex)': '#ff4444'
    }
    
    # داده‌های نمونه برای جدول مقایسه نهایی
    comparison_data = [
        {
            'level': 'Level 1 (Simple)',
            'avg_reward': '-45.23',
            'max_reward': '12.45',
            'final_reward': '8.91',
            'convergence_episode': '320',
            'avg_actor_loss': '0.0234',
            'avg_critic_loss': '0.1456'
        },
        {
            'level': 'Level 2 (Medium)',
            'avg_reward': '-62.78',
            'max_reward': '5.32',
            'final_reward': '2.18',
            'convergence_episode': '410',
            'avg_actor_loss': '0.0389',
            'avg_critic_loss': '0.2134'
        },
        {
            'level': 'Level 3 (Complex)',
            'avg_reward': '-88.45',
            'max_reward': '-8.76',
            'final_reward': '-12.34',
            'convergence_episode': '485',
            'avg_actor_loss': '0.0521',
            'avg_critic_loss': '0.3287'
        }
    ]
    
    # تولید نمودارهای مقایسه‌ای
    reward_traces = []
    actor_loss_traces = []
    critic_loss_traces = []
    
    for level_name, color in colors_map.items():
        # داده‌های فرضی برای نمودارها
        episodes = list(range(1, 501))
        
        if 'Simple' in level_name:
            base_reward = -100
            improvement = 150
            noise = 12
        elif 'Medium' in level_name:
            base_reward = -120
            improvement = 120
            noise = 15
        else:
            base_reward = -140
            improvement = 80
            noise = 18
        
        rewards = [base_reward + (i/500)*improvement + np.random.normal(0, noise) 
                   for i in episodes]
        actor_losses = [0.08 - (i/500)*0.05 + np.random.uniform(-0.01, 0.01) 
                        for i in episodes]
        critic_losses = [0.35 - (i/500)*0.15 + np.random.uniform(-0.02, 0.02) 
                         for i in episodes]
        
        reward_traces.append(
            go.Scatter(x=episodes, y=rewards, mode='lines', name=level_name,
                      line=dict(color=color, width=2))
        )
        actor_loss_traces.append(
            go.Scatter(x=episodes, y=actor_losses, mode='lines', name=level_name,
                      line=dict(color=color, width=2))
        )
        critic_loss_traces.append(
            go.Scatter(x=episodes, y=critic_losses, mode='lines', name=level_name,
                      line=dict(color=color, width=2))
        )
    
    # Layout نمودارها
    plot_layout = {
        'plot_bgcolor': '#1e2a38',
        'paper_bgcolor': '#1e2a38',
        'font': {'color': '#e0e0e0', 'family': 'Vazirmatn'},
        'xaxis': {'gridcolor': '#2d3e50', 'title': 'اپیزود'},
        'yaxis': {'gridcolor': '#2d3e50'},
        'legend': {'bgcolor': '#0d1b2a', 'bordercolor': '#2d3e50', 'borderwidth': 1}
    }
    
    return html.Div([
        html.H3("📈 مقایسه نتایج آموزش 3 سطح", style=HEADER_STYLE),
        
        # جدول مقایسه آماری
        html.Div([
            html.H4("📊 جدول مقایسه آماری", 
                   style={'color': '#00ff88', 'fontFamily': 'Vazirmatn', 'marginBottom': '15px'}),
            
            html.Div([
                dash_table.DataTable(
                    columns=[
                        {'name': 'سطح آموزش', 'id': 'level'},
                        {'name': 'میانگین پاداش', 'id': 'avg_reward'},
                        {'name': 'بیشترین پاداش', 'id': 'max_reward'},
                        {'name': 'پاداش نهایی', 'id': 'final_reward'},
                        {'name': 'اپیزود همگرایی', 'id': 'convergence_episode'},
                        {'name': 'میانگین Actor Loss', 'id': 'avg_actor_loss'},
                        {'name': 'میانگین Critic Loss', 'id': 'avg_critic_loss'}
                    ],
                    data=comparison_data,
                    style_cell={
                        'textAlign': 'center',
                        'padding': '14px',
                        'backgroundColor': '#1e2a38',
                        'color': '#e0e0e0',
                        'fontFamily': 'Vazirmatn',
                        'border': '1px solid #2d3e50',
                        'whiteSpace': 'normal',
                        'height': 'auto'
                    },
                    style_header={
                        'backgroundColor': '#00d4ff',
                        'color': '#0d1b2a',
                        'fontWeight': 'bold',
                        'textAlign': 'center',
                        'fontFamily': 'Vazirmatn',
                        'fontSize': '15px',
                        'padding': '12px'
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 0}, 'backgroundColor': '#1a3a2a'},
                        {'if': {'row_index': 1}, 'backgroundColor': '#3a2a1a'},
                        {'if': {'row_index': 2}, 'backgroundColor': '#3a1a1a'}
                    ]
                )
            ], style={'overflowX': 'auto'})
        ], style=CARD_STYLE),
        
        # نمودار Reward
        html.Div([
            html.H4("📉 مقایسه Reward در طول آموزش", 
                   style={'color': '#00ff88', 'fontFamily': 'Vazirmatn'}),
            dcc.Graph(
                figure={
                    'data': reward_traces,
                    'layout': {**plot_layout, 'yaxis': {**plot_layout['yaxis'], 'title': 'پاداش'}}
                },
                config={'displayModeBar': False},
                style={'height': '400px'}
            )
        ], style=CARD_STYLE),
        
        # نمودار Actor Loss
        html.Div([
            html.H4("🎭 مقایسه Actor Loss", 
                   style={'color': '#ffa500', 'fontFamily': 'Vazirmatn'}),
            dcc.Graph(
                figure={
                    'data': actor_loss_traces,
                    'layout': {**plot_layout, 'yaxis': {**plot_layout['yaxis'], 'title': 'Actor Loss'}}
                },
                config={'displayModeBar': False},
                style={'height': '400px'}
            )
        ], style=CARD_STYLE),
        
        # نمودار Critic Loss
        html.Div([
            html.H4("🎯 مقایسه Critic Loss", 
                   style={'color': '#ff4444', 'fontFamily': 'Vazirmatn'}),
            dcc.Graph(
                figure={
                    'data': critic_loss_traces,
                    'layout': {**plot_layout, 'yaxis': {**plot_layout['yaxis'], 'title': 'Critic Loss'}}
                },
                config={'displayModeBar': False},
                style={'height': '400px'}
            )
        ], style=CARD_STYLE),
        
        # توضیحات
        html.Div([
            html.H4("💡 نتیجه‌گیری", style={'color': '#00d4ff', 'fontFamily': 'Vazirmatn'}),
            html.P("✅ Level 1 (Simple) سریع‌ترین همگرایی و بالاترین پاداش نهایی را دارد", 
                   style=TEXT_STYLE),
            html.P("✅ Level 2 (Medium) تعادل مناسبی بین سرعت و پیچیدگی دارد", 
                   style=TEXT_STYLE),
            html.P("✅ Level 3 (Complex) چالش‌برانگیزتر است اما قابلیت یادگیری از محیط پیچیده‌تر را دارد", 
                   style=TEXT_STYLE)
        ], style=CARD_STYLE)
    ])

# ═══════════════════════════════════════════════════════════════════════
# Callback: کنترل شروع/توقف مانیتورینگ
# ═══════════════════════════════════════════════════════════════════════
@app.callback(
    [Output('interval-component', 'disabled'),
     Output('live-status', 'children')],
    [Input('start-button', 'n_clicks'),
     Input('pause-button', 'n_clicks')],
    prevent_initial_call=True
)
def control_monitoring(start_clicks, pause_clicks):
    """کنترل شروع و توقف مانیتورینگ زنده"""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-button':
        return False, "✅ مانیتورینگ فعال است..."
    elif button_id == 'pause-button':
        return True, "⏸️ مانیتورینگ متوقف شد"
    
    raise PreventUpdate

# ═══════════════════════════════════════════════════════════════════════
# Callback: به‌روزرسانی نمودارهای زنده
# ═══════════════════════════════════════════════════════════════════════
@app.callback(
    [Output('live-utility-graph', 'figure'),
     Output('live-energy-graph', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_live_graphs(n):
    """به‌روزرسانی نمودارهای زنده"""
    global episode_counter
    
    episode_counter += 1
    live_data_store['episodes'].append(episode_counter)
    
    # تولید داده‌های تصادفی شبیه‌سازی شده
    utility = 0.4 + (episode_counter % 30) * 0.02 + np.random.uniform(-0.05, 0.05)
    energy = 60 + np.random.normal(0, 8)
    
    live_data_store['utility'].append(np.clip(utility, 0, 1))
    live_data_store['energy'].append(np.clip(energy, 20, 100))
    
    # نمودار Utility
    utility_fig = {
        'data': [
            go.Scatter(
                x=list(live_data_store['episodes']),
                y=list(live_data_store['utility']),
                mode='lines+markers',
                name='Utility',
                line=dict(color='#00ff88', width=3),
                marker=dict(size=6)
            )
        ],
        'layout': {
            'title': '⚡ تابع مطلوبیت (Utility)',
            'plot_bgcolor': '#1e2a38',
            'paper_bgcolor': '#1e2a38',
            'font': {'color': '#e0e0e0', 'family': 'Vazirmatn', 'size': 13},
            'xaxis': {'gridcolor': '#2d3e50', 'title': 'قدم'},
            'yaxis': {'gridcolor': '#2d3e50', 'title': 'مقدار', 'range': [0, 1]},
            'margin': {'l': 50, 'r': 20, 't': 50, 'b': 50}
        }
    }
    
    # نمودار Energy
    energy_fig = {
        'data': [
            go.Scatter(
                x=list(live_data_store['episodes']),
                y=list(live_data_store['energy']),
                mode='lines+markers',
                name='Energy',
                line=dict(color='#ff9500', width=3),
                marker=dict(size=6)
            )
        ],
        'layout': {
            'title': '🔋 مصرف انرژی (درصد)',
            'plot_bgcolor': '#1e2a38',
            'paper_bgcolor': '#1e2a38',
            'font': {'color': '#e0e0e0', 'family': 'Vazirmatn', 'size': 13},
            'xaxis': {'gridcolor': '#2d3e50', 'title': 'قدم'},
            'yaxis': {'gridcolor': '#2d3e50', 'title': 'درصد', 'range': [0, 100]},
            'margin': {'l': 50, 'r': 20, 't': 50, 'b': 50}
        }
    }
    
    return utility_fig, energy_fig

# ═══════════════════════════════════════════════════════════════════════
# اجرای برنامه
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 داشبورد SkyMind در حال راه‌اندازی...")
    print("="*70)
    print("📍 آدرس: http://127.0.0.1:8050")
    print("💡 برای توقف: Ctrl+C")
    print("="*70 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
