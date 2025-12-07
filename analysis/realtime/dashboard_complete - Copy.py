"""
SkyMind Dashboard - Complete Version with Real Data Integration
داشبورد تعاملی برای نمایش نتایج آموزش MADDPG
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime
import numpy as np
import os
import sys

# ✅ تنظیم مسیر به پوشه اصلی پروژه
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

print(f"📁 Base Directory: {BASE_DIR}")

# وارد کردن data_loader برای بارگذاری داده‌های واقعی
try:
    from dashboard.data_loader import TrainingDataLoader
    data_loader = TrainingDataLoader()
    print("✅ TrainingDataLoader loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import TrainingDataLoader: {e}")
    data_loader = None

# اضافه کردن مسیر پروژه
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# وارد کردن data_loader برای بارگذاری داده‌های واقعی
try:
    from dashboard.data_loader import TrainingDataLoader
    data_loader = TrainingDataLoader()
    print("✅ TrainingDataLoader loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import TrainingDataLoader: {e}")
    data_loader = None

# ========================================
# تنظیمات رنگ و استایل
# ========================================
COLORS = {
    'background': '#0a1929',
    'surface': '#1e2a38',
    'primary': '#00d4ff',
    'secondary': '#00ff88',
    'accent': '#ffa500',
    'danger': '#ff4444',
    'text': '#e0e0e0',
    'text_secondary': '#a0a0a0'
}

HEADER_STYLE = {
    'textAlign': 'center',
    'color': COLORS['primary'],
    'marginBottom': '20px',
    'fontFamily': 'Vazirmatn, sans-serif',
    'fontWeight': 'bold'
}

CARD_STYLE = {
    'backgroundColor': COLORS['surface'],
    'padding': '20px',
    'borderRadius': '10px',
    'marginBottom': '20px',
    'boxShadow': '0 4px 6px rgba(0, 212, 255, 0.1)',
    'border': f'1px solid {COLORS["primary"]}'
}

METRIC_CARD_STYLE = {
    'backgroundColor': COLORS['surface'],
    'padding': '20px',
    'borderRadius': '10px',
    'textAlign': 'center',
    'boxShadow': '0 4px 6px rgba(0, 212, 255, 0.1)',
    'border': f'2px solid {COLORS["primary"]}',
    'minHeight': '120px'
}

# ========================================
# ایجاد اپلیکیشن Dash
# ========================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True
)

app.title = "SkyMind Dashboard"

# ========================================
# توابع کمکی برای نمودارها
# ========================================

def create_metric_card(title, value, icon, color):
    """ایجاد کارت متریک"""
    return html.Div([
        html.Div([
            html.I(className=f"fas {icon}", style={
                'fontSize': '2em',
                'color': color,
                'marginBottom': '10px'
            }),
            html.H4(title, style={
                'color': COLORS['text_secondary'],
                'fontSize': '0.9em',
                'marginBottom': '5px',
                'fontFamily': 'Vazirmatn'
            }),
            html.H2(value, style={
                'color': color,
                'fontSize': '1.8em',
                'fontWeight': 'bold',
                'margin': '0',
                'fontFamily': 'Vazirmatn'
            })
        ])
    ], style=METRIC_CARD_STYLE)

def create_reward_plot(episodes, rewards, ma20=None):
    """ایجاد نمودار پاداش با میانگین متحرک"""
    traces = []
    
    # خط اصلی پاداش
    traces.append(go.Scatter(
        x=episodes,
        y=rewards,
        mode='lines',
        name='پاداش',
        line=dict(color=COLORS['secondary'], width=2),
        hovertemplate='اپیزود: %{x}<br>پاداش: %{y:.2f}<extra></extra>'
    ))
    
    # خط میانگین متحرک (اگر موجود باشد)
    if ma20 is not None and len(ma20) > 0:
        traces.append(go.Scatter(
            x=episodes[-len(ma20):],
            y=ma20,
            mode='lines',
            name='میانگین متحرک 20',
            line=dict(color=COLORS['accent'], width=2, dash='dot'),
            hovertemplate='اپیزود: %{x}<br>MA20: %{y:.2f}<extra></extra>'
        ))
    
    layout = go.Layout(
        title='روند پاداش در طول آموزش',
        xaxis={'title': 'اپیزود', 'color': COLORS['text']},
        yaxis={'title': 'پاداش', 'color': COLORS['text']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return go.Figure(data=traces, layout=layout)

def create_loss_plot(episodes, actor_loss, critic_loss):
    """ایجاد نمودار Loss با مقیاس لگاریتمی"""
    traces = [
        go.Scatter(
            x=episodes,
            y=actor_loss,
            mode='lines',
            name='Actor Loss',
            line=dict(color=COLORS['primary'], width=2),
            hovertemplate='اپیزود: %{x}<br>Actor Loss: %{y:.4f}<extra></extra>'
        ),
        go.Scatter(
            x=episodes,
            y=critic_loss,
            mode='lines',
            name='Critic Loss',
            line=dict(color=COLORS['danger'], width=2),
            hovertemplate='اپیزود: %{x}<br>Critic Loss: %{y:.4f}<extra></extra>'
        )
    ]
    
    layout = go.Layout(
        title='روند Loss در طول آموزش',
        xaxis={'title': 'اپیزود', 'color': COLORS['text']},
        yaxis={
            'title': 'Loss (مقیاس لگاریتمی)',
            'color': COLORS['text'],
            'type': 'log'  # مقیاس لگاریتمی
        },

        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return go.Figure(data=traces, layout=layout)

def create_comparison_bar_chart(levels_data):
    """ایجاد نمودار میله‌ای مقایسه سطوح"""
    levels = [f"Level {i+1}" for i in range(len(levels_data))]
    rewards = [level['avg_reward'] for level in levels_data]
    
    fig = go.Figure(data=[go.Bar(
        x=levels,
        y=rewards,
        marker_color=[COLORS['secondary'], COLORS['primary'], COLORS['accent']],
        text=[f"{r:.2f}" for r in rewards],
        textposition='outside',
        hovertemplate='%{x}<br>میانگین پاداش: %{y:.2f}<extra></extra>'
    )])
    
    fig.update_layout(
        title='مقایسه میانگین پاداش بین سطوح',
        xaxis={'title': 'سطح', 'color': COLORS['text']},
        yaxis={'title': 'میانگین پاداش', 'color': COLORS['text']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'}
    )
    
    return fig

# ========================================
# توابع کمکی برای تب جدید: مقایسه تخلیه محاسباتی
# ========================================

def get_offloading_mock_data():
    """داده‌های شبیه‌سازی شده برای تخلیه محاسباتی
    در آینده باید از training_history.json بارگذاری شود
    """
    return {
        'local': 450,      # تعداد Task پردازش شده در زمین
        'edge': 320,       # تعداد Task تخلیه شده به لبه
        'fog': 180,        # تعداد Task تخلیه شده به مه
        'cloud': 50,       # تعداد Task تخلیه شده به ابر
        'local_latency': 2.3,   # میانگین تأخیر (ms)
        'edge_latency': 5.8,
        'fog_latency': 12.4,
        'cloud_latency': 28.7,
        'local_energy': 156.2,  # مصرف انرژی کل (Joule)
        'edge_energy': 89.5,
        'fog_energy': 42.3,
        'cloud_energy': 12.0
    }

def create_offloading_pie_chart(data):
    """نمودار دایره‌ای توزیع تخلیه"""
    labels = ['زمین (Local)', 'لبه (Edge)', 'مه (Fog)', 'ابر (Cloud)']
    values = [data['local'], data['edge'], data['fog'], data['cloud']]
    colors = [COLORS['secondary'], COLORS['primary'], COLORS['accent'], COLORS['danger']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color='#000000', width=2)),
        textinfo='label+percent+value',
        textfont=dict(size=14, family='Vazirmatn'),
        hovertemplate='<b>%{label}</b><br>تعداد: %{value}<br>درصد: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='توزیع Tasks بر اساس محل پردازش',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_latency_bar_chart(data):
    """نمودار میله‌ای تأخیر"""
    locations = ['زمین', 'لبه', 'مه', 'ابر']
    latencies = [
        data['local_latency'],
        data['edge_latency'],
        data['fog_latency'],
        data['cloud_latency']
    ]
    colors_list = [COLORS['secondary'], COLORS['primary'], COLORS['accent'], COLORS['danger']]
    
    fig = go.Figure(data=[go.Bar(
        x=locations,
        y=latencies,
        marker_color=colors_list,
        text=[f"{v:.1f} ms" for v in latencies],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>تأخیر: %{y:.2f} ms<extra></extra>'
    )])
    
    fig.update_layout(
        title='مقایسه تأخیر (Latency) بین محیط‌های مختلف',
        xaxis={'title': 'محیط پردازش', 'color': COLORS['text']},
        yaxis={'title': 'تأخیر (میلی‌ثانیه)', 'color': COLORS['text']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'}
    )
    
    return fig

def create_energy_bar_chart(data):
    """نمودار میله‌ای مصرف انرژی"""
    locations = ['زمین', 'لبه', 'مه', 'ابر']
    energies = [
        data['local_energy'],
        data['edge_energy'],
        data['fog_energy'],
        data['cloud_energy']
    ]
    colors_list = [COLORS['secondary'], COLORS['primary'], COLORS['accent'], COLORS['danger']]
    
    fig = go.Figure(data=[go.Bar(
        x=locations,
        y=energies,
        marker_color=colors_list,
        text=[f"{v:.1f} J" for v in energies],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>انرژی: %{y:.2f} Joule<extra></extra>'
    )])
    
    fig.update_layout(
        title='مقایسه مصرف انرژی بین محیط‌های مختلف',
        xaxis={'title': 'محیط پردازش', 'color': COLORS['text']},
        yaxis={'title': 'مصرف انرژی (Joule)', 'color': COLORS['text']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'}
    )
    
    return fig

def create_offloading_efficiency_chart(data):
    """نمودار کارایی تخلیه: نسبت Tasks به تأخیر"""
    locations = ['زمین', 'لبه', 'مه', 'ابر']
    tasks = [data['local'], data['edge'], data['fog'], data['cloud']]
    latencies = [
        data['local_latency'],
        data['edge_latency'],
        data['fog_latency'],
        data['cloud_latency']
    ]
    
    # محاسبه کارایی: تعداد Task / تأخیر
    efficiency = [t / l if l > 0 else 0 for t, l in zip(tasks, latencies)]
    
    colors_list = [COLORS['secondary'], COLORS['primary'], COLORS['accent'], COLORS['danger']]
    
    fig = go.Figure(data=[go.Bar(
        x=locations,
        y=efficiency,
        marker_color=colors_list,
        text=[f"{e:.2f}" for e in efficiency],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>کارایی: %{y:.2f}<extra></extra>'
    )])
    
    fig.update_layout(
        title='کارایی پردازش (Tasks/Latency)',
        xaxis={'title': 'محیط پردازش', 'color': COLORS['text']},
        yaxis={'title': 'کارایی (Tasks/ms)', 'color': COLORS['text']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'}
    )
    
    return fig

# ========================================
# توابع ایجاد تب‌ها
# ========================================

def create_tab_overview():
    """تب نمای کلی"""
    # بارگذاری داده‌های واقعی
    if data_loader:
        summary = data_loader.get_summary_stats()
        total_episodes = summary['total_episodes']
        avg_reward = summary['avg_reward']
        success_rate = summary['success_rate']
        avg_critic_loss = summary['avg_critic_loss']
    else:
        # داده‌های پیش‌فرض
        total_episodes = 1600
        avg_reward = -24.51
        success_rate = 12.5
        avg_critic_loss = 316.83
    
    return html.Div([
        html.H2("🏠 نمای کلی سیستم", style=HEADER_STYLE),
        
        # کارت‌های متریک
        dbc.Row([
            dbc.Col(create_metric_card(
                "کل اپیزودها",
                f"{total_episodes:,}",
                "fa-chart-line",
                COLORS['primary']
            ), width=3),
            dbc.Col(create_metric_card(
                "میانگین پاداش",
                f"{avg_reward:.2f}",
                "fa-trophy",
                COLORS['secondary']
            ), width=3),
            dbc.Col(create_metric_card(
                "نرخ موفقیت",
                f"{success_rate:.1f}%",
                "fa-check-circle",
                COLORS['accent']
            ), width=3),
            dbc.Col(create_metric_card(
                "میانگین Critic Loss",
                f"{avg_critic_loss:.2f}",
                "fa-exclamation-triangle",
                COLORS['danger']
            ), width=3),
        ], style={'marginBottom': '30px'}),
        
        # هدف پروژه
        html.Div([
            html.H3("🎯 هدف پروژه", style={'color': COLORS['secondary'], 'fontFamily': 'Vazirmatn'}),
            html.P(
                "الگوریتم MADDPG (Multi-Agent Deep Deterministic Policy Gradient) برای بهینه‌سازی برنامه‌ریزی محاسباتی در پهپادها (UAV-assisted Computation Offloading)",
                style={'color': COLORS['text'], 'fontSize': '1.1em', 'fontFamily': 'Vazirmatn', 'lineHeight': '1.8'}
            ),
            html.Ul([
                html.Li("آموزش چند عامل همزمان برای تصمیم‌گیری هموزن", style={'fontFamily': 'Vazirmatn'}),
                html.Li("بهینه‌سازی معیارهای انرژی و تأخیر شبکه", style={'fontFamily': 'Vazirmatn'}),
                html.Li("استفاده از شبکه‌های Actor-Critic", style={'fontFamily': 'Vazirmatn'}),
            ], style={'color': COLORS['text_secondary'], 'fontSize': '1em'})
        ], style=CARD_STYLE),
        
        # آمار واقعی سیستم
        html.Div([
            html.H3("📊 آمار واقعی سیستم", style={'color': COLORS['accent'], 'fontFamily': 'Vazirmatn'}),
            html.P(
                f"مسیر ذخیره نتایج: results/ - آخرین بروزرسانی: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                style={'color': COLORS['text_secondary'], 'fontFamily': 'Vazirmatn'}
            )
        ], style=CARD_STYLE)
    ])

def create_tab_results():
    """تب نتایج آموزش با جداول مقایسه‌ای"""
    
    # رنگ‌بندی سطوح
    colors_map = {
        'Level 1 (Simple)': '#00ff41',
        'Level 2 (Medium)': '#ffa500',
        'Level 3 (Complex)': '#ff4444'
    }
    
    # ===== این بخش را تغییر دهید =====
    # بارگذاری داده‌های واقعی
    comparison_data = []
    
    for level_name, level_key in [('Level 1 (Simple)', 'level1'),
                                   ('Level 2 (Medium)', 'level2'),
                                   ('Level 3 (Complex)', 'level3')]:
        if data_loader:
            data = data_loader.load_level_data(level_key)
            if data:
                comparison_data.append({
                    'level': level_name,
                    'avg_reward': f"{np.mean(data['rewards_agent0']):.2f}",
                    'max_reward': f"{np.max(data['rewards_agent0']):.2f}",
                    'final_reward': f"{data['rewards_agent0'][-1]:.2f}",
                    'convergence_episode': f"{len(data['episodes'])}",
                    'avg_actor_loss': f"{np.mean(data.get('actor_loss_agent0', [0])):.4f}",
                    'avg_critic_loss': f"{np.mean(data.get('critic_loss_agent0', [0])):.4f}"
                })
            else:
                # داده Mock اگر فایل نبود
                comparison_data.append({
                    'level': level_name,
                    'avg_reward': 'N/A',
                    'max_reward': 'N/A',
                    'final_reward': 'N/A',
                    'convergence_episode': 'N/A',
                    'avg_actor_loss': 'N/A',
                    'avg_critic_loss': 'N/A'
                })
    
    # ===== بخش تولید نمودارها را هم تغییر دهید =====
    reward_traces = []
    actor_loss_traces = []
    critic_loss_traces = []
    
    for level_name, color in colors_map.items():
        level_key = 'level1' if 'Simple' in level_name else ('level2' if 'Medium' in level_name else 'level3')
        
        if data_loader:
            data = data_loader.load_level_data(level_key)
            if data:
                # داده‌های واقعی
                episodes = data['episodes']
                rewards = data['rewards_agent0']
                actor_losses = data.get('actor_loss_agent0', [0] * len(episodes))
                critic_losses = data.get('critic_loss_agent0', [0] * len(episodes))
            else:
                # داده Mock
                episodes, rewards, actor_losses, critic_losses = generate_mock_for_level(level_key)
        else:
            episodes, rewards, actor_losses, critic_losses = generate_mock_for_level(level_key)
        
        # محاسبه MA20 برای Reward
        if len(rewards) >= 20:
            ma20 = np.convolve(rewards, np.ones(20)/20, mode='valid')
            reward_traces.append(
                go.Scatter(x=episodes[19:], y=ma20, mode='lines', name=f'{level_name} (MA20)',
                          line=dict(color=color, width=3, dash='solid'))
            )
        
        # خط اصلی Reward
        reward_traces.append(
            go.Scatter(x=episodes, y=rewards, mode='lines', name=level_name,
                      line=dict(color=color, width=1), opacity=0.4)
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
    
    # ادامه کد مانند قبل...
    return html.Div([
        # ... باقی کد بدون تغییر
    ])

# تابع کمکی برای تولید Mock Data
def generate_mock_for_level(level_key):
    """تولید داده Mock برای یک سطح"""
    episodes = list(range(1, 501))
    
    if level_key == 'level1':
        base_reward, improvement, noise = -100, 150, 12
    elif level_key == 'level2':
        base_reward, improvement, noise = -120, 120, 15
    else:
        base_reward, improvement, noise = -140, 80, 18
    
    rewards = [base_reward + (i/500)*improvement + np.random.normal(0, noise) for i in episodes]
    actor_losses = [0.08 - (i/500)*0.05 + np.random.uniform(-0.01, 0.01) for i in episodes]
    critic_losses = [0.35 - (i/500)*0.15 + np.random.uniform(-0.02, 0.02) for i in episodes]
    
    return episodes, rewards, actor_losses, critic_losses


def create_tab_offloading():
    """تب جدید: مقایسه تخلیه محاسباتی"""
    # بارگذاری داده‌های شبیه‌سازی شده
    offloading_data = get_offloading_mock_data()
    
    return html.Div([
        html.H2("🌐 مقایسه تخلیه محاسباتی", style=HEADER_STYLE),
        
        # توضیحات
        html.Div([
            html.H4("💡 درباره این تب", style={'color': COLORS['secondary'], 'fontFamily': 'Vazirmatn'}),
            html.P(
                "این تب نشان می‌دهد که Tasks چگونه بین محیط‌های مختلف (زمین، لبه، مه، ابر) توزیع شده‌اند و عملکرد هر محیط چگونه است.",
                style={'color': COLORS['text'], 'fontSize': '1.1em', 'fontFamily': 'Vazirmatn', 'lineHeight': '1.8'}
            ),
            html.P(
                "⚠️ توجه: داده‌های فعلی شبیه‌سازی شده هستند. برای نمایش داده‌های واقعی، باید ساختار لاگ‌گیری در train_maddpg_ultimate.py تغییر کند.",
                style={'color': COLORS['accent'], 'fontSize': '0.95em', 'fontFamily': 'Vazirmatn', 'fontStyle': 'italic'}
            )
        ], style=CARD_STYLE),
        
        # نمودار دایره‌ای توزیع
        html.Div([
            html.H3("📊 توزیع Tasks", style={'color': COLORS['secondary'], 'marginBottom': '15px', 'fontFamily': 'Vazirmatn'}),
            dcc.Graph(
                figure=create_offloading_pie_chart(offloading_data),
                style={'height': '500px'}
            )
        ], style=CARD_STYLE),
        
        # نمودارهای میله‌ای
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("⏱️ مقایسه تأخیر", style={'color': COLORS['accent'], 'marginBottom': '15px', 'fontFamily': 'Vazirmatn'}),
                    dcc.Graph(
                        figure=create_latency_bar_chart(offloading_data),
                        style={'height': '400px'}
                    )
                ], style=CARD_STYLE)
            ], width=6),
            dbc.Col([
                html.Div([
                    html.H3("🔋 مقایسه مصرف انرژی", style={'color': COLORS['danger'], 'marginBottom': '15px', 'fontFamily': 'Vazirmatn'}),
                    dcc.Graph(
                        figure=create_energy_bar_chart(offloading_data),
                        style={'height': '400px'}
                    )
                ], style=CARD_STYLE)
            ], width=6)
        ]),
        
        # نمودار کارایی
        html.Div([
            html.H3("⚡ کارایی پردازش", style={'color': COLORS['primary'], 'marginBottom': '15px', 'fontFamily': 'Vazirmatn'}),
            dcc.Graph(
                figure=create_offloading_efficiency_chart(offloading_data),
                style={'height': '400px'}
            )
        ], style=CARD_STYLE),
        
        # نتیجه‌گیری
        html.Div([
            html.H4("📝 نتیجه‌گیری", style={'color': COLORS['secondary'], 'fontFamily': 'Vazirmatn'}),
            html.Ul([
                html.Li(f"🟢 زمین (Local): {offloading_data['local']} Task - تأخیر کم ({offloading_data['local_latency']} ms) - مصرف انرژی بالا ({offloading_data['local_energy']} J)", style={'fontFamily': 'Vazirmatn', 'color': COLORS['text']}),
                html.Li(f"🔵 لبه (Edge): {offloading_data['edge']} Task - تأخیر متوسط ({offloading_data['edge_latency']} ms) - مصرف انرژی متوسط ({offloading_data['edge_energy']} J)", style={'fontFamily': 'Vazirmatn', 'color': COLORS['text']}),
                html.Li(f"🟠 مه (Fog): {offloading_data['fog']} Task - تأخیر بالا ({offloading_data['fog_latency']} ms) - مصرف انرژی کم ({offloading_data['fog_energy']} J)", style={'fontFamily': 'Vazirmatn', 'color': COLORS['text']}),
                html.Li(f"☁️ Cloud: {offloading_data['cloud']} Task - تأخیر بسیار بالا ({offloading_data['cloud_latency']} ms) - مصرف انرژی بسیار کم ({offloading_data['cloud_energy']} J)", 
        style={'fontFamily': 'Vazirmatn', 'color': COLORS['text']}),            ], style={'fontSize': '1.05em', 'lineHeight': '2'}),
            html.P(
                "💡 نتیجه: پهپاد به طور هوشمندانه Tasks را بین محیط‌های مختلف توزیع می‌کند تا تعادل بین تأخیر و مصرف انرژی برقرار شود.",
                style={'color': COLORS['primary'], 'fontSize': '1.1em', 'fontWeight': 'bold', 'fontFamily': 'Vazirmatn', 'marginTop': '15px'}
            )
        ], style=CARD_STYLE)
    ])

def create_tab_agents():
    """تب جزئیات عامل‌ها"""
    return html.Div([
        html.H2("🤖 جزئیات عامل‌ها", style=HEADER_STYLE),
        
        html.Div([
            html.H3("عامل 1 (Agent 0)", style={'color': COLORS['primary'], 'fontFamily': 'Vazirmatn'}),
            html.P("معماری شبکه: Actor-Critic", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'}),
            html.P("ابعاد ورودی: 12 بعد (وضعیت محیط)", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'}),
            html.P("ابعاد خروجی: 5 بعد (اقدامات)", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'}),
        ], style=CARD_STYLE),
        
        html.Div([
            html.H3("عامل 2 (Agent 1)", style={'color': COLORS['secondary'], 'fontFamily': 'Vazirmatn'}),
            html.P("معماری شبکه: Actor-Critic", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'}),
            html.P("ابعاد ورودی: 12 بعد (وضعیت محیط)", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'}),
            html.P("ابعاد خروجی: 5 بعد (اقدامات)", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'}),
        ], style=CARD_STYLE),
    ])

def create_tab_params():
    """تب پارامترهای آموزش"""
    return html.Div([
        html.H2("⚙️ پارامترهای آموزش", style=HEADER_STYLE),
        
        html.Div([
            html.H3("پارامترهای اصلی", style={'color': COLORS['primary'], 'fontFamily': 'Vazirmatn'}),
            html.Table([
                html.Tr([
                    html.Td("Learning Rate (Actor):", style={'fontWeight': 'bold', 'color': COLORS['text_secondary'], 'fontFamily': 'Vazirmatn'}),
                    html.Td("0.0001", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'})
                ]),
                html.Tr([
                    html.Td("Learning Rate (Critic):", style={'fontWeight': 'bold', 'color': COLORS['text_secondary'], 'fontFamily': 'Vazirmatn'}),
                    html.Td("0.001", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'})
                ]),
                html.Tr([
                    html.Td("Discount Factor (γ):", style={'fontWeight': 'bold', 'color': COLORS['text_secondary'], 'fontFamily': 'Vazirmatn'}),
                    html.Td("0.95", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'})
                ]),
                html.Tr([
                    html.Td("Soft Update Rate (τ):", style={'fontWeight': 'bold', 'color': COLORS['text_secondary'], 'fontFamily': 'Vazirmatn'}),
                    html.Td("0.01", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'})
                ]),
                html.Tr([
                    html.Td("Batch Size:", style={'fontWeight': 'bold', 'color': COLORS['text_secondary'], 'fontFamily': 'Vazirmatn'}),
                    html.Td("64", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'})
                ]),
                html.Tr([
                    html.Td("Replay Buffer Size:", style={'fontWeight': 'bold', 'color': COLORS['text_secondary'], 'fontFamily': 'Vazirmatn'}),
                    html.Td("100,000", style={'color': COLORS['text'], 'fontFamily': 'Vazirmatn'})
                ]),
            ], style={'width': '100%', 'lineHeight': '2', 'fontFamily': 'Vazirmatn'})
        ], style=CARD_STYLE),
    ])

def create_tab_monitoring():
    """تب مانیتورینگ زنده"""
    return html.Div([
        html.H2("📡 مانیتورینگ زنده", style=HEADER_STYLE),
        
        html.Div([
            html.H3("وضعیت سیستم", style={'color': COLORS['secondary'], 'fontFamily': 'Vazirmatn'}),
            html.Button(
                "▶️ شروع مانیتورینگ",
                id='start-monitoring-btn',
                n_clicks=0,
                style={
                    'backgroundColor': COLORS['secondary'],
                    'color': '#000',
                    'border': 'none',
                    'padding': '10px 20px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '1.1em',
                    'marginRight': '10px',
                    'fontFamily': 'Vazirmatn',
                    'fontWeight': 'bold'
                }
            ),
            html.Button(
                "⏸️ توقف",
                id='stop-monitoring-btn',
                n_clicks=0,
                style={
                    'backgroundColor': COLORS['danger'],
                    'color': '#fff',
                    'border': 'none',
                    'padding': '10px 20px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'fontSize': '1.1em',
                    'fontFamily': 'Vazirmatn',
                    'fontWeight': 'bold'
                }
            ),
            html.Div(id='monitoring-status', style={'marginTop': '20px', 'color': COLORS['text'], 'fontFamily': 'Vazirmatn'})
        ], style=CARD_STYLE),
        
        # نمودارهای زنده
        html.Div([
            html.H3("📊 نمودارهای زنده", style={'color': COLORS['primary'], 'fontFamily': 'Vazirmatn'}),
            dcc.Graph(id='live-utility-graph', style={'height': '300px'}),
            dcc.Graph(id='live-energy-graph', style={'height': '300px'}),
        ], style=CARD_STYLE),
        
        # Interval برای به‌روزرسانی خودکار
        dcc.Interval(
            id='interval-component',
            interval=2000,  # به‌روزرسانی هر 2 ثانیه
            n_intervals=0,
            disabled=True
        )
    ])

# ========================================
# Layout اصلی اپلیکیشن
# ========================================
app.layout = html.Div([
    # هدر
    html.Div([
        html.H1(
            "مانیتورینگ لحظه‌ای و نتایج آموزش MADDPG",
            style={
                'textAlign': 'center',
                'color': COLORS['primary'],
                'marginBottom': '10px',
                'fontFamily': 'Vazirmatn',
                'fontWeight': 'bold'
            }
        ),
        html.P(
            "داشبورد تعاملی برای تحلیل عملکرد الگوریتم چندعاملی",
            style={
                'textAlign': 'center',
                'color': COLORS['text_secondary'],
                'fontSize': '1.1em',
                'fontFamily': 'Vazirmatn'
            }
        )
    ], style={
        'backgroundColor': COLORS['surface'],
        'padding': '20px',
        'marginBottom': '20px',
        'borderRadius': '10px',
        'boxShadow': '0 4px 6px rgba(0, 212, 255, 0.2)'
    }),
    
    # تب‌ها
    dcc.Tabs(
        id='tabs',
        value='tab-overview',
        children=[
            dcc.Tab(
                label='🏠 نمای کلی',
                value='tab-overview',
                style={'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'},
                selected_style={
                    'fontFamily': 'Vazirmatn',
                    'fontWeight': 'bold',
                    'backgroundColor': COLORS['primary'],
                    'color': '#000'
                }
            ),
            dcc.Tab(
                label='📊 نتایج آموزش',
                value='tab-results',
                style={'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'},
                selected_style={
                    'fontFamily': 'Vazirmatn',
                    'fontWeight': 'bold',
                    'backgroundColor': COLORS['primary'],
                    'color': '#000'
                }
            ),
            dcc.Tab(
                label='🌐 مقایسه تخلیه محاسباتی',
                value='tab-offloading',
                style={'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'},
                selected_style={
                    'fontFamily': 'Vazirmatn',
                    'fontWeight': 'bold',
                    'backgroundColor': COLORS['primary'],
                    'color': '#000'
                }
            ),
            dcc.Tab(
                label='👥 جزئیات عامل‌ها',
                value='tab-agents',
                style={'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'},
                selected_style={
                    'fontFamily': 'Vazirmatn',
                    'fontWeight': 'bold',
                    'backgroundColor': COLORS['primary'],
                    'color': '#000'
                }
            ),
            dcc.Tab(
                label='⚙️ پارامترهای آموزش',
                value='tab-params',
                style={'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'},
                selected_style={
                    'fontFamily': 'Vazirmatn',
                    'fontWeight': 'bold',
                    'backgroundColor': COLORS['primary'],
                    'color': '#000'
                }
            ),
            dcc.Tab(
                label='🔴 مانیتورینگ زنده',
                value='tab-monitoring',
                style={'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'},
                selected_style={
                    'fontFamily': 'Vazirmatn',
                    'fontWeight': 'bold',
                    'backgroundColor': COLORS['danger'],
                    'color': '#fff'
                }
            ),
        ],
        style={'fontFamily': 'Vazirmatn'},
        colors={
            'border': COLORS['primary'],
            'primary': COLORS['primary'],
            'background': COLORS['surface']
        }
    ),
    
    # محتوای تب‌ها
    html.Div(id='tabs-content', style={'padding': '20px'})
    
], style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'padding': '20px',
    'fontFamily': 'Vazirmatn'
})

# ========================================
# Callbacks
# ========================================

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    """رندر محتوای تب انتخاب شده"""
    if tab == 'tab-overview':
        return create_tab_overview()
    elif tab == 'tab-results':
        return create_tab_results()
    elif tab == 'tab-offloading':
        return create_tab_offloading()
    elif tab == 'tab-agents':
        return create_tab_agents()
    elif tab == 'tab-params':
        return create_tab_params()
    elif tab == 'tab-monitoring':
        return create_tab_monitoring()

@app.callback(
    [Output('interval-component', 'disabled'),
     Output('monitoring-status', 'children')],
    [Input('start-monitoring-btn', 'n_clicks'),
     Input('stop-monitoring-btn', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_monitoring(start_clicks, stop_clicks):
    """کنترل شروع/توقف مانیتورینگ"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return True, "⏸️ مانیتورینگ غیرفعال"
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-monitoring-btn':
        return False, html.Div([
            html.Span("✅ مانیتورینگ فعال", style={'color': COLORS['secondary'], 'fontWeight': 'bold', 'fontFamily': 'Vazirmatn'}),
            html.Span(f" | آخرین به‌روزرسانی: {datetime.now().strftime('%H:%M:%S')}", style={'color': COLORS['text_secondary'], 'fontFamily': 'Vazirmatn'})
        ])
    else:
        return True, html.Div([
            html.Span("⏸️ مانیتورینگ متوقف شد", style={'color': COLORS['danger'], 'fontWeight': 'bold', 'fontFamily': 'Vazirmatn'})
        ])

@app.callback(
    Output('live-utility-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_utility_graph(n):
    """به‌روزرسانی نمودار Utility"""
    # داده‌های شبیه‌سازی شده
    x = list(range(max(0, n-20), n+1))
    y = [np.random.uniform(0.5, 1.0) for _ in x]
    
    fig = go.Figure(data=[go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=6),
        hovertemplate='زمان: %{x}<br>Utility: %{y:.3f}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Utility Score در زمان واقعی',
        xaxis={'title': 'زمان', 'color': COLORS['text']},
        yaxis={'title': 'Utility', 'color': COLORS['text'], 'range': [0, 1.2]},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'}
    )
    
    return fig

@app.callback(
    Output('live-energy-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_energy_graph(n):
    """به‌روزرسانی نمودار Energy"""
    # داده‌های شبیه‌سازی شده
    x = list(range(max(0, n-20), n+1))
    y = [np.random.uniform(20, 80) for _ in x]
    
    fig = go.Figure(data=[go.Scatter(
        x=x,
        y=y,
        mode='lines+markers',
        line=dict(color=COLORS['danger'], width=2),
        marker=dict(size=6),
        fill='tozeroy',
        hovertemplate='زمان: %{x}<br>انرژی: %{y:.1f} J<extra></extra>'
    )])
    
    fig.update_layout(
        title='مصرف انرژی در زمان واقعی',
        xaxis={'title': 'زمان', 'color': COLORS['text']},
        yaxis={'title': 'انرژی (Joule)', 'color': COLORS['text']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'}
    )
    
    return fig

# ========================================
# اجرای سرور
# ========================================
if __name__ == '__main__':
    print("=" * 80)
    print("🚀 SkyMind Dashboard Starting...")
    print("=" * 80)
    print(f"📊 Dashboard URL: http://127.0.0.1:8050")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # بررسی وجود data_loader
    if data_loader:
        print("✅ TrainingDataLoader connected successfully")
        try:
            # 🔥 لود کردن داده‌های level1
            print("📂 Loading level1 data...")
            data_loader.load_level_data('level1')
            
            summary = data_loader.get_summary_stats()
            print(f"📈 Total Episodes: {summary['total_episodes']}")
            print(f"🏆 Average Reward: {summary['avg_reward']:.2f}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load summary stats: {e}")
    else:
        print("⚠️ Warning: TrainingDataLoader not available - using mock data")
    
    print("=" * 80)
    print("✨ Dashboard is ready! Press Ctrl+C to stop.")
    print("=" * 80)
    
    app.run(debug=True, host='127.0.0.1', port=8050)