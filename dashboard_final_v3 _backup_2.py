import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ========================================
# تنظیمات اولیه
# ========================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "SkyMind Dashboard v3.1"

COLORS = {
    'background': '#0a0e27',
    'surface': '#1a1f3a',
    'surface_light': '#2a2f4a',
    'primary': '#00d4ff',
    'secondary': '#b794f6',
    'success': '#48bb78',
    'warning': '#ed8936',
    'danger': '#f56565',
    'text': '#e2e8f0',
    'text_secondary': '#a0aec0'
}

TRANSLATIONS = {
    'fa': {
        'title': 'داشبورد تحلیل SkyMind',
        'subtitle': 'سیستم هوشمند تخلیه وظایف با MADDPG',
        'overview': 'نمای کلی',
        'training': 'نتایج آموزش',
        'layer': 'تحلیل لایه‌ای',
        'final_sr': 'نرخ موفقیت نهایی',
        'best_reward': 'بهترین پاداش',
        'convergence': 'Episode همگرایی'
    },
    'en': {
        'title': 'SkyMind Analysis Dashboard',
        'subtitle': 'Intelligent Task Offloading with MADDPG',
        'overview': 'Overview',
        'training': 'Training Results',
        'layer': 'Layer Analysis',
        'final_sr': 'Final Success Rate',
        'best_reward': 'Best Reward',
        'convergence': 'Convergence Episode'
    }
}

# ========================================
# داده‌های پروژه
# ========================================

# نتایج نهایی (فصل 4)
FINAL_RESULTS = {
    'success_rate': 95.0,
    'best_reward': 130.53,
    'convergence_episode': 842,
    'final_battery': 3.82,
    'final_latency': 54.23,
    'final_overload': 12.89,
    'total_episodes': 1000,
    'training_hours': 18.5
}

# State Space (432-dim) - فصل 3
STATE_SPACE = {
    'uav_state': 10,  # موقعیت، سرعت، باتری، CPU، صف
    'global_graph': 256,  # GNN embedding
    'neighbor_attention': 80,  # 4 همسایه × 20-dim
    'task_features': 40,  # 20 task × 2-dim
    'channel_state': 20,  # کیفیت کانال
    'edge_server_state': 26  # 2 سرور × 13-dim
}

# Action Space - فصل 3
ACTION_SPACE = {
    'discrete': {
        'name': 'Offload Decision',
        'size': 5,
        'options': ['Local', 'Terrestrial Edge', 'Aerial Edge', 'Cloud', 'Reject']
    },
    'continuous': {
        'name': 'Heuristic Parameters',
        'size': 6,
        'params': ['CPU Freq', 'Bandwidth', 'Movement X', 'Movement Y', 'Queue Priority', 'Energy Mode']
    }
}

# Architecture (616K params) - فصل 3
ARCHITECTURE = {
    'actor': {
        'input': 432,
        'hidden': [512, 512, 256],
        'output': 11,  # 5 discrete + 6 continuous
        'params': 616000,
        'activation': 'ELU + LayerNorm'
    },
    'critic': {
        'input': 487,  # 432 state + 55 actions (5 UAVs × 11)
        'hidden': [512, 256, 128],
        'output': 1,
        'activation': 'ELU'
    }
}

# Hyperparameters - فصل 3
HYPERPARAMS = {
    'batch_size': 256,
    'lr_actor': 1e-4,
    'lr_critic': 1e-3,
    'gamma': 0.95,
    'tau': 0.01,
    'buffer_size': 100000,
    'epsilon_start': 0.9,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.995
}

# Reward Components - فصل 3
REWARD_COMPONENTS = {
    'latency': {'weight': -0.4, 'range': '[0, 200] ms'},
    'energy': {'weight': -0.3, 'range': '[0, 1] normalized'},
    'overload': {'weight': -0.2, 'range': '[0, 1] probability'},
    'success': {'weight': +0.1, 'range': '{0, 1} binary'}
}

# محیط‌های آموزشی (جدول 4.7)
COMPLEXITY_ENVIRONMENTS = {
    'Easy': {'stage': 1, 'obstacles': 0, 'episodes': '0-1000'},
    'Medium': {'stage': 2, 'obstacles': 2, 'episodes': '1000-2500'},
    'Complex': {'stage': 3, 'obstacles': 4, 'episodes': '2500-4000'}
}

# جدول 4.6: عملکرد بر اساس سطح پیچیدگی
COMPLEXITY_PERFORMANCE = {
    'Easy': {
        'battery': 0.0382,
        'latency': 54.23,
        'overload': 0.1289,
        'success_rate': 97.2,
        'initial_reward': -17.67,
        'final_reward': 12.34,
        'best_reward': 18.52,
        'convergence_episode': 250,
        'actor_loss': 2.87,
        'critic_loss': 0.082,
        'training_hours': 18
    },
    'Medium': {
        'battery': 0.0425,
        'latency': 62.48,
        'overload': 0.1567,
        'success_rate': 95.4,
        'initial_reward': -23.45,
        'final_reward': 3.67,
        'best_reward': 9.23,
        'convergence_episode': 380,
        'actor_loss': 3.42,
        'critic_loss': 0.127,
        'training_hours': 22
    },
    'Complex': {
        'battery': 0.0489,
        'latency': 71.35,
        'overload': 0.1893,
        'success_rate': 93.1,
        'initial_reward': -35.82,
        'final_reward': -8.91,
        'best_reward': -2.14,
        'convergence_episode': 450,
        'actor_loss': 4.15,
        'critic_loss': 0.198,
        'training_hours': 28
    }
}

# جدول 4.13: توزیع انتخاب لایه در سطوح مختلف
LAYER_DISTRIBUTION_COMPLEXITY = {
    'Easy': {
        'local': 28.3,
        'terrestrial_edge': 52.3,
        'aerial_edge': 12.8,
        'cloud': 6.6
    },
    'Medium': {
        'local': 24.5,
        'terrestrial_edge': 38.7,
        'aerial_edge': 24.2,
        'cloud': 12.6
    },
    'Complex': {
        'local': 18.2,
        'terrestrial_edge': 20.4,
        'aerial_edge': 55.8,
        'cloud': 5.6
    }
}

# جدول 4.1: استراتژی‌های Heuristic
HEURISTIC_STRATEGIES = {
    'Conservative': {
        'battery': 0.0382,
        'latency': 58.12,
        'overload': 0.1156,
        'success_rate': 97.8,
        'reward': 125.34,
        'description': 'اولویت به کاهش مصرف انرژی'
    },
    'Balanced': {
        'battery': 0.0401,
        'latency': 54.23,
        'overload': 0.1289,
        'success_rate': 95.0,
        'reward': 130.53,
        'description': 'تعادل بین تمام معیارها'
    },
    'Adaptive': {
        'battery': 0.0425,
        'latency': 52.87,
        'overload': 0.1401,
        'success_rate': 94.2,
        'reward': 128.91,
        'description': 'تطبیق پویا با شرایط'
    },
    'Greedy': {
        'battery': 0.0489,
        'latency': 49.34,
        'overload': 0.1678,
        'success_rate': 91.5,
        'reward': 118.76,
        'description': 'اولویت به کاهش تأخیر'
    }
}

# جدول 4.2: مقایسه Ablation
ABLATION_RESULTS = {
    'Full Model': {
        'reward': 130.53,
        'final_avg': 12.34,
        'cohens_d': 0.0,
        'p_value': 1.0,
        'significance': '—'
    },
    'No GAT': {
        'reward': 95.24,
        'final_avg': -20.24,
        'cohens_d': 0.3774,
        'p_value': 8.57e-3,
        'significance': '⭐'
    },
    'No Temporal': {
        'reward': 118.63,
        'final_avg': -26.63,
        'cohens_d': -0.0758,
        'p_value': 5.94e-1,
        'significance': '—'
    },
    'Decentralized': {
        'reward': 65.81,
        'final_avg': -85.81,
        'cohens_d': 0.4923,
        'p_value': 6.52e-4,
        'significance': '⭐⭐'
    },
    'Simpler Arch': {
        'reward': 45.69,
        'final_avg': -82.69,
        'cohens_d': 1.1250,
        'p_value': 1.72e-13,
        'significance': '⭐⭐⭐'
    }
}

# Baseline Methods
BASELINE_METHODS = {
    'MADDPG (Ours)': {
        'battery': 3.82,
        'latency': 54.23,
        'success_rate': 95.0,
        'reward': 130.53
    },
    'Random': {
        'battery': 8.91,
        'latency': 125.67,
        'success_rate': 45.2,
        'reward': -245.32
    },
    'Always Local': {
        'battery': 9.45,
        'latency': 89.34,
        'success_rate': 62.8,
        'reward': -89.23
    },
    'Always Edge': {
        'battery': 4.23,
        'latency': 78.56,
        'success_rate': 78.5,
        'reward': 45.67
    },
    'Round Robin': {
        'battery': 5.67,
        'latency': 92.14,
        'success_rate': 71.3,
        'reward': 12.89
    },
    'Load Balance': {
        'battery': 4.89,
        'latency': 68.92,
        'success_rate': 82.4,
        'reward': 78.45
    }
}

# ========================================
# توابع نمودارسازی - بخش Overview
# ========================================

def create_metrics_gauge(value, title, max_val, color, format_str='%'):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': COLORS['text']}},
        delta={'reference': max_val * 0.8, 'increasing': {'color': color}},
        number={'suffix': format_str, 'font': {'size': 32}},
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': color},
            'bar': {'color': color},
            'bgcolor': COLORS['surface_light'],
            'borderwidth': 2,
            'bordercolor': color,
            'steps': [
                {'range': [0, max_val*0.33], 'color': COLORS['surface']},
                {'range': [max_val*0.33, max_val*0.66], 'color': COLORS['surface_light']}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor=COLORS['surface'],
        plot_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=250,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def create_state_space_sunburst(lang):
    labels = ['State (432)', 'UAV (10)', 'Graph (256)', 'Attention (80)', 'Task (40)', 'Channel (20)', 'Edge (26)']
    parents = ['', 'State (432)', 'State (432)', 'State (432)', 'State (432)', 'State (432)', 'State (432)']
    values = [432, 10, 256, 80, 40, 20, 26]
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=[COLORS['primary'], COLORS['success'], COLORS['warning'], 
                    COLORS['danger'], COLORS['secondary'], COLORS['primary'], COLORS['success']],
            line=dict(color=COLORS['background'], width=2)
        ),
        textfont=dict(size=14, color='white', family='Vazirmatn')
    ))
    
    fig.update_layout(
        title={'text': '🧠 ساختار فضای حالت (432-بعدی)', 'font': {'size': 18, 'color': COLORS['primary']}},
        paper_bgcolor=COLORS['surface'],
        plot_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        margin=dict(t=60, l=0, r=0, b=0)
    )
    return fig

def create_action_space_chart(lang):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('🎯 Discrete (5 Classes)', '⚙️ Continuous (6 Params)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Discrete
    discrete_options = ACTION_SPACE['discrete']['options']
    discrete_values = [20, 35, 15, 10, 20]
    fig.add_trace(go.Bar(
        x=discrete_options,
        y=discrete_values,
        marker_color=[COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger'], COLORS['secondary']],
        text=[f"{v}%" for v in discrete_values],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    
    # Continuous
    continuous_params = ['CPU', 'BW', 'Move-X', 'Move-Y', 'Queue', 'Energy']
    continuous_ranges = [2.5, 100, 20, 20, 50, 500]
    fig.add_trace(go.Bar(
        x=continuous_params,
        y=continuous_ranges,
        marker_color=COLORS['secondary'],
        text=[f"{v}" for v in continuous_ranges],
        textposition='outside',
        showlegend=False
    ), row=1, col=2)
    
    fig.update_layout(
        title={'text': '🎮 فضای عمل ترکیبی (Hybrid)', 'font': {'size': 18, 'color': COLORS['primary']}},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor=COLORS['surface_light'])
    fig.update_yaxes(gridcolor=COLORS['surface_light'])
    
    return fig

# ========================================
# توابع نمودارسازی - بخش Training
# ========================================

def create_learning_curve(lang):
    episodes = np.linspace(0, 1000, 500)
    rewards = -50 + 180 / (1 + np.exp(-(episodes - 500) / 150))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes,
        y=rewards,
        mode='lines',
        line=dict(color=COLORS['primary'], width=3),
        fill='tozeroy',
        fillcolor=f"rgba(0, 212, 255, 0.1)",
        name='Average Reward'
    ))
    
    fig.add_hline(y=130.53, line_dash="dash", line_color=COLORS['success'], 
                  annotation_text="Best: +130.53", annotation_position="right")
    
    fig.update_layout(
        title={'text': '📈 منحنی یادگیری (1000 Episode)', 'font': {'size': 20, 'color': COLORS['primary']}},
        xaxis={'title': 'Episode', 'gridcolor': COLORS['surface_light']},
        yaxis={'title': 'Average Reward', 'gridcolor': COLORS['surface_light']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500,
        hovermode='x unified'
    )
    return fig

def create_loss_curves(lang):
    episodes = np.linspace(0, 1000, 200)
    actor_loss = 10 * np.exp(-episodes / 250) + 2
    critic_loss = 5 * np.exp(-episodes / 200) + 0.5
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes, y=actor_loss, mode='lines',
        name='Actor Loss', line=dict(color=COLORS['primary'], width=2)
    ))
    fig.add_trace(go.Scatter(
        x=episodes, y=critic_loss, mode='lines',
        name='Critic Loss', line=dict(color=COLORS['secondary'], width=2)
    ))
    
    fig.update_layout(
        title={'text': '📉 منحنی‌های Loss', 'font': {'size': 18, 'color': COLORS['primary']}},
        xaxis={'title': 'Episode', 'gridcolor': COLORS['surface_light']},
        yaxis={'title': 'Loss Value', 'type': 'log', 'gridcolor': COLORS['surface_light']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        hovermode='x unified'
    )
    return fig

# ========================================
# توابع نمودارسازی - بخش Layer Analysis
# ========================================

def create_layer_distribution_pie(lang):
    labels = ['Local', 'Terrestrial Edge', 'Aerial Edge', 'Cloud']
    values = [22.3, 45.8, 25.4, 6.5]
    colors_list = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors_list, line=dict(color=COLORS['background'], width=2)),
        textposition='outside',
        textinfo='label+percent',
        textfont=dict(size=14, color='white', family='Vazirmatn')
    )])
    
    fig.update_layout(
        title={'text': '🌐 توزیع انتخاب لایه‌ها', 'font': {'size': 18, 'color': COLORS['primary']}},
        paper_bgcolor=COLORS['surface'],
        plot_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

# ========================================
# توابع نمودارسازی - بخش Heuristics
# ========================================

def create_heuristic_comparison_radar(lang):
    categories = ['Battery', 'Latency', 'Overload', 'Success Rate']
    
    fig = go.Figure()
    
    for strategy, data in HEURISTIC_STRATEGIES.items():
        values = [
            100 - (data['battery'] / 0.05) * 100,  # Inverse for battery
            100 - (data['latency'] / 100) * 100,    # Inverse for latency
            100 - (data['overload'] / 0.2) * 100,   # Inverse for overload
            data['success_rate']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=strategy,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title={'text': '⚡ مقایسه استراتژی‌های Heuristic (جدول 4.1)', 'font': {'size': 18, 'color': COLORS['primary']}},
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=COLORS['surface_light']),
            angularaxis=dict(gridcolor=COLORS['surface_light'])
        ),
        paper_bgcolor=COLORS['surface'],
        plot_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500,
        showlegend=True
    )
    return fig

def create_heuristic_metrics_bars(lang):
    strategies = list(HEURISTIC_STRATEGIES.keys())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🔋 مصرف باتری (%)', '⏱️ تأخیر (ms)', '✅ نرخ موفقیت (%)', '🎯 پاداش کل'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Battery
    battery_vals = [HEURISTIC_STRATEGIES[s]['battery'] * 100 for s in strategies]
    fig.add_trace(go.Bar(x=strategies, y=battery_vals, marker_color=COLORS['danger'],
                         text=[f"{v:.2f}" for v in battery_vals], textposition='outside', showlegend=False), 
                  row=1, col=1)
    
    # Latency
    latency_vals = [HEURISTIC_STRATEGIES[s]['latency'] for s in strategies]
    fig.add_trace(go.Bar(x=strategies, y=latency_vals, marker_color=COLORS['primary'],
                         text=[f"{v:.1f}" for v in latency_vals], textposition='outside', showlegend=False),
                  row=1, col=2)
    
    # Success Rate
    sr_vals = [HEURISTIC_STRATEGIES[s]['success_rate'] for s in strategies]
    fig.add_trace(go.Bar(x=strategies, y=sr_vals, marker_color=COLORS['success'],
                         text=[f"{v:.1f}" for v in sr_vals], textposition='outside', showlegend=False),
                  row=2, col=1)
    
    # Reward
    reward_vals = [HEURISTIC_STRATEGIES[s]['reward'] for s in strategies]
    fig.add_trace(go.Bar(x=strategies, y=reward_vals, marker_color=COLORS['warning'],
                         text=[f"{v:.1f}" for v in reward_vals], textposition='outside', showlegend=False),
                  row=2, col=2)
    
    fig.update_layout(
        title={'text': '📊 معیارهای عملکرد استراتژی‌ها', 'font': {'size': 18, 'color': COLORS['primary']}},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor=COLORS['surface_light'])
    fig.update_yaxes(gridcolor=COLORS['surface_light'])
    
    return fig

# ========================================
# توابع نمودارسازی - بخش Ablation
# ========================================

def create_ablation_comparison_chart(lang):
    variants = list(ABLATION_RESULTS.keys())
    rewards = [ABLATION_RESULTS[v]['reward'] for v in variants]
    colors_map = [COLORS['success'], COLORS['danger'], COLORS['warning'], COLORS['danger'], COLORS['danger']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=variants,
        y=rewards,
        marker_color=colors_map,
        text=[f"{r:.1f}" for r in rewards],
        textposition='outside',
        textfont=dict(size=14, color='white')
    ))
    
    fig.add_hline(y=130.53, line_dash="dash", line_color=COLORS['success'],
                  annotation_text="Full Model: 130.53", annotation_position="right")
    
    fig.update_layout(
        title={'text': '🔬 مقایسه Ablation Study (جدول 4.2)', 'font': {'size': 20, 'color': COLORS['primary']}},
        xaxis={'title': 'Variant', 'gridcolor': COLORS['surface_light']},
        yaxis={'title': 'Best Reward', 'gridcolor': COLORS['surface_light']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500
    )
    return fig

# ========================================
# توابع نمودارسازی - بخش Baseline
# ========================================

def create_baseline_comparison_chart(lang):
    methods = list(BASELINE_METHODS.keys())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🔋 مصرف باتری (%)', '⏱️ تأخیر (ms)', '✅ نرخ موفقیت (%)', '🎯 پاداش'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Battery
    battery_vals = [BASELINE_METHODS[m]['battery'] for m in methods]
    fig.add_trace(go.Bar(x=methods, y=battery_vals, marker_color=COLORS['danger'],
                         text=[f"{v:.2f}" for v in battery_vals], textposition='outside', showlegend=False),
                  row=1, col=1)
    
    # Latency
    latency_vals = [BASELINE_METHODS[m]['latency'] for m in methods]
    fig.add_trace(go.Bar(x=methods, y=latency_vals, marker_color=COLORS['primary'],
                         text=[f"{v:.1f}" for v in latency_vals], textposition='outside', showlegend=False),
                  row=1, col=2)
    
    # Success Rate
    sr_vals = [BASELINE_METHODS[m]['success_rate'] for m in methods]
    fig.add_trace(go.Bar(x=methods, y=sr_vals, marker_color=COLORS['success'],
                         text=[f"{v:.1f}" for v in sr_vals], textposition='outside', showlegend=False),
                  row=2, col=1)
    
    # Reward
    reward_vals = [BASELINE_METHODS[m]['reward'] for m in methods]
    colors_reward = [COLORS['success'] if r > 0 else COLORS['danger'] for r in reward_vals]
    fig.add_trace(go.Bar(x=methods, y=reward_vals, marker_color=colors_reward,
                         text=[f"{v:.1f}" for v in reward_vals], textposition='outside', showlegend=False),
                  row=2, col=2)
    
    fig.update_layout(
        title={'text': '📉 مقایسه با روش‌های Baseline', 'font': {'size': 18, 'color': COLORS['primary']}},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor=COLORS['surface_light'])
    fig.update_yaxes(gridcolor=COLORS['surface_light'])
    
    return fig

# ========================================
# توابع نمودارسازی - بخش Complexity Analysis 🆕
# ========================================

def create_complexity_metrics_comparison(lang):
    """نمودار مقایسه 4 معیار در 3 سطح"""
    levels = list(COMPLEXITY_PERFORMANCE.keys())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🔋 مصرف باتری (%)', '⏱️ تأخیر (ms)', '⚠️ احتمال سربار (%)', '✅ نرخ موفقیت (%)'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Battery
    battery_values = [COMPLEXITY_PERFORMANCE[l]['battery'] * 100 for l in levels]
    fig.add_trace(go.Scatter(
        x=levels, y=battery_values, 
        mode='lines+markers+text',
        line=dict(color=COLORS['danger'], width=4),
        marker=dict(size=12, symbol='circle'),
        text=[f"{v:.2f}%" for v in battery_values],
        textposition='top center',
        name='Battery'
    ), row=1, col=1)
    
    # Latency
    latency_values = [COMPLEXITY_PERFORMANCE[l]['latency'] for l in levels]
    fig.add_trace(go.Scatter(
        x=levels, y=latency_values,
        mode='lines+markers+text',
        line=dict(color=COLORS['primary'], width=4),
        marker=dict(size=12, symbol='square'),
        text=[f"{v:.1f}" for v in latency_values],
        textposition='top center',
        name='Latency'
    ), row=1, col=2)
    
    # Overload
    overload_values = [COMPLEXITY_PERFORMANCE[l]['overload'] * 100 for l in levels]
    fig.add_trace(go.Scatter(
        x=levels, y=overload_values,
        mode='lines+markers+text',
        line=dict(color=COLORS['warning'], width=4),
        marker=dict(size=12, symbol='diamond'),
        text=[f"{v:.2f}%" for v in overload_values],
        textposition='top center',
        name='Overload'
    ), row=2, col=1)
    
    # Success Rate
    sr_values = [COMPLEXITY_PERFORMANCE[l]['success_rate'] for l in levels]
    fig.add_trace(go.Scatter(
        x=levels, y=sr_values,
        mode='lines+markers+text',
        line=dict(color=COLORS['success'], width=4),
        marker=dict(size=12, symbol='triangle-up'),
        text=[f"{v:.1f}%" for v in sr_values],
        textposition='bottom center',
        name='Success Rate'
    ), row=2, col=2)
    
    fig.update_layout(
        title={'text': '📊 تأثیر پیچیدگی محیط بر عملکرد (جدول 4.6)', 'font': {'size': 20, 'color': COLORS['primary']}},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor=COLORS['surface_light'])
    fig.update_yaxes(gridcolor=COLORS['surface_light'])
    
    return fig

def create_learning_curves_complexity(lang):
    """منحنی‌های یادگیری شبیه‌سازی شده"""
    episodes = np.linspace(0, 4000, 200)
    
    fig = go.Figure()
    
    colors_map = {'Easy': COLORS['success'], 'Medium': COLORS['warning'], 'Complex': COLORS['danger']}
    
    for level, data in COMPLEXITY_PERFORMANCE.items():
        initial = data['initial_reward']
        final = data['final_reward']
        convergence = data['convergence_episode']
        
        # منحنی سیگموید
        rewards = initial + (final - initial) / (1 + np.exp(-(episodes - convergence) / 300))
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=rewards,
            mode='lines',
            name=f"{level} (همگرایی: {convergence})",
            line=dict(color=colors_map[level], width=3)
        ))
    
    fig.update_layout(
        title={'text': '📈 منحنی‌های یادگیری در سطوح پیچیدگی (جدول 4.8)', 'font': {'size': 20, 'color': COLORS['primary']}},
        xaxis={'title': 'Episode', 'gridcolor': COLORS['surface_light']},
        yaxis={'title': 'Average Reward', 'gridcolor': COLORS['surface_light']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_layer_distribution_stacked(lang):
    """نمودار Stacked Bar توزیع لایه‌ها"""
    levels = list(LAYER_DISTRIBUTION_COMPLEXITY.keys())
    
    fig = go.Figure()
    
    layers = ['local', 'terrestrial_edge', 'aerial_edge', 'cloud']
    layer_names = ['Local', 'Terrestrial Edge', 'Aerial Edge', 'Cloud']
    colors_list = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]
    
    for i, (layer, name) in enumerate(zip(layers, layer_names)):
        values = [LAYER_DISTRIBUTION_COMPLEXITY[level][layer] for level in levels]
        fig.add_trace(go.Bar(
            name=name,
            x=levels,
            y=values,
            marker_color=colors_list[i],
            text=[f"{v:.1f}%" for v in values],
            textposition='inside'
        ))
    
    fig.update_layout(
        title={'text': '🌐 توزیع انتخاب لایه در سطوح مختلف (جدول 4.13)', 'font': {'size': 20, 'color': COLORS['primary']}},
        xaxis={'title': 'سطح پیچیدگی', 'gridcolor': COLORS['surface_light']},
        yaxis={'title': 'درصد (%)', 'gridcolor': COLORS['surface_light']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        barmode='stack',
        height=500
    )
    
    return fig

def create_training_cost_chart(lang):
    """نمودار هزینه محاسباتی"""
    levels = list(COMPLEXITY_PERFORMANCE.keys())
    
    hours = [COMPLEXITY_PERFORMANCE[l]['training_hours'] for l in levels]
    convergence = [COMPLEXITY_PERFORMANCE[l]['convergence_episode'] for l in levels]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('⏰ زمان آموزش (ساعت)', '🎯 Episode همگرایی'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(go.Bar(
        x=levels, y=hours,
        marker_color=[COLORS['success'], COLORS['warning'], COLORS['danger']],
        text=[f"{h}h" for h in hours],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=levels, y=convergence,
        marker_color=[COLORS['success'], COLORS['warning'], COLORS['danger']],
        text=convergence,
        textposition='outside',
        showlegend=False
    ), row=1, col=2)
    
    fig.update_layout(
        title={'text': '💰 هزینه محاسباتی آموزش (جدول 4.7)', 'font': {'size': 20, 'color': COLORS['primary']}},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400
    )
    
    fig.update_xaxes(gridcolor=COLORS['surface_light'])
    fig.update_yaxes(gridcolor=COLORS['surface_light'])
    
    return fig

def create_loss_comparison_complexity(lang):
    """مقایسه Loss در سطوح مختلف"""
    levels = list(COMPLEXITY_PERFORMANCE.keys())
    
    actor_loss = [COMPLEXITY_PERFORMANCE[l]['actor_loss'] for l in levels]
    critic_loss = [COMPLEXITY_PERFORMANCE[l]['critic_loss'] for l in levels]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Actor Loss',
        x=levels,
        y=actor_loss,
        marker_color=COLORS['primary'],
        text=[f"{v:.2f}" for v in actor_loss],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Critic Loss',
        x=levels,
        y=critic_loss,
        marker_color=COLORS['secondary'],
        text=[f"{v:.3f}" for v in critic_loss],
        textposition='outside'
    ))
    
    fig.update_layout(
        title={'text': '📉 مقایسه Actor/Critic Loss', 'font': {'size': 18, 'color': COLORS['primary']}},
        xaxis={'title': 'سطح پیچیدگی', 'gridcolor': COLORS['surface_light']},
        yaxis={'title': 'Loss Value', 'gridcolor': COLORS['surface_light']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        barmode='group',
        height=400
    )
    
    return fig
# ========================================
# توابع رندر تب‌ها
# ========================================

def render_overview_tab(lang, t):
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🎯 نتایج نهایی", className="text-center"), 
                                   style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['primary']}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H3(f"{FINAL_RESULTS['success_rate']}%", className="text-success text-center"),
                                html.P("نرخ موفقیت", className="text-center text-secondary")
                            ], width=4),
                            dbc.Col([
                                html.H3(f"+{FINAL_RESULTS['best_reward']}", className="text-warning text-center"),
                                html.P("بهترین پاداش", className="text-center text-secondary")
                            ], width=4),
                            dbc.Col([
                                html.H3(f"{FINAL_RESULTS['convergence_episode']}", className="text-primary text-center"),
                                html.P("Episode همگرایی", className="text-center text-secondary")
                            ], width=4),
                        ])
                    ])
                ], style={'backgroundColor': COLORS['surface'], 'marginBottom': '20px'})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_metrics_gauge(FINAL_RESULTS['final_battery'], 
                                                           '🔋 مصرف باتری', 10, COLORS['danger'], '%'))], width=4),
            dbc.Col([dcc.Graph(figure=create_metrics_gauge(FINAL_RESULTS['final_latency'], 
                                                           '⏱️ تأخیر', 100, COLORS['primary'], ' ms'))], width=4),
            dbc.Col([dcc.Graph(figure=create_metrics_gauge(FINAL_RESULTS['final_overload'], 
                                                           '⚠️ احتمال سربار', 20, COLORS['warning'], '%'))], width=4),
        ], style={'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_state_space_sunburst(lang))], width=6),
            dbc.Col([dcc.Graph(figure=create_action_space_chart(lang))], width=6),
        ], style={'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🧠 معماری شبکه‌های عصبی", className="text-center"),
                                   style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['primary']}),
                    dbc.CardBody([
                        html.Div([
                            html.H6("Actor Network:", className="text-info"),
                            html.P(f"Input: {ARCHITECTURE['actor']['input']}-dim → Hidden: {ARCHITECTURE['actor']['hidden']} → Output: {ARCHITECTURE['actor']['output']}-dim"),
                            html.P(f"Parameters: {ARCHITECTURE['actor']['params']:,} | Activation: {ARCHITECTURE['actor']['activation']}"),
                            html.Hr(style={'borderColor': COLORS['surface_light']}),
                            html.H6("Critic Network:", className="text-warning"),
                            html.P(f"Input: {ARCHITECTURE['critic']['input']}-dim → Hidden: {ARCHITECTURE['critic']['hidden']} → Output: {ARCHITECTURE['critic']['output']}-dim"),
                            html.P(f"Activation: {ARCHITECTURE['critic']['activation']}"),
                        ], style={'fontFamily': 'Vazirmatn', 'color': COLORS['text']})
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("⚙️ هایپرپارامترها", className="text-center"),
                                   style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['primary']}),
                    dbc.CardBody([
                        html.Div([
                            html.P(f"Batch Size: {HYPERPARAMS['batch_size']}"),
                            html.P(f"Learning Rate (Actor): {HYPERPARAMS['lr_actor']}"),
                            html.P(f"Learning Rate (Critic): {HYPERPARAMS['lr_critic']}"),
                            html.P(f"Gamma (Discount): {HYPERPARAMS['gamma']}"),
                            html.P(f"Tau (Soft Update): {HYPERPARAMS['tau']}"),
                            html.P(f"Buffer Size: {HYPERPARAMS['buffer_size']:,}"),
                            html.P(f"Epsilon: {HYPERPARAMS['epsilon_start']} → {HYPERPARAMS['epsilon_end']} (decay: {HYPERPARAMS['epsilon_decay']})")
                        ], style={'fontFamily': 'Vazirmatn', 'color': COLORS['text']})
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=6),
        ])
    ], fluid=True)


def render_training_tab(lang, t):
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_learning_curve(lang))], width=12)
        ], style={'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_loss_curves(lang))], width=12)
        ], style={'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🎁 مؤلفه‌های تابع پاداش", className="text-center"),
                                   style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['primary']}),
                    dbc.CardBody([
                        html.Div([
                            html.P(f"⏱️ Latency: وزن {REWARD_COMPONENTS['latency']['weight']} | دامنه: {REWARD_COMPONENTS['latency']['range']}"),
                            html.P(f"🔋 Energy: وزن {REWARD_COMPONENTS['energy']['weight']} | دامنه: {REWARD_COMPONENTS['energy']['range']}"),
                            html.P(f"⚠️ Overload: وزن {REWARD_COMPONENTS['overload']['weight']} | دامنه: {REWARD_COMPONENTS['overload']['range']}"),
                            html.P(f"✅ Success: وزن {REWARD_COMPONENTS['success']['weight']} | دامنه: {REWARD_COMPONENTS['success']['range']}"),
                            html.Hr(style={'borderColor': COLORS['surface_light']}),
                            html.P("فرمول نهایی: R = -0.4×Latency - 0.3×Energy - 0.2×Overload + 0.1×Success", 
                                   className="text-info font-weight-bold")
                        ], style={'fontFamily': 'Vazirmatn', 'color': COLORS['text']})
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=12)
        ])
    ], fluid=True)


def render_layer_tab(lang, t):
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_layer_distribution_pie(lang))], width=12)
        ], style={'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 تحلیل توزیع لایه‌ها", className="text-center"),
                                   style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['primary']}),
                    dbc.CardBody([
                        html.Div([
                            html.H6("🟢 Local Processing (22.3%):", className="text-success"),
                            html.P("وظایف سبک با تأخیر بسیار کم اما محدودیت CPU"),
                            html.Hr(style={'borderColor': COLORS['surface_light']}),
                            
                            html.H6("🔵 Terrestrial Edge (45.8%):", className="text-primary"),
                            html.P("بیشترین انتخاب - تعادل بین تأخیر، انرژی و قابلیت پردازش"),
                            html.Hr(style={'borderColor': COLORS['surface_light']}),
                            
                            html.H6("🟠 Aerial Edge (25.4%):", className="text-warning"),
                            html.P("برای مناطق بدون پوشش زمینی یا وظایف متحرک"),
                            html.Hr(style={'borderColor': COLORS['surface_light']}),
                            
                            html.H6("🔴 Cloud (6.5%):", className="text-danger"),
                            html.P("وظایف بسیار سنگین که نیاز به منابع زیاد دارند (با تأخیر بالا)")
                        ], style={'fontFamily': 'Vazirmatn', 'color': COLORS['text']})
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=12)
        ])
    ], fluid=True)


def render_heuristics_tab(lang, t):
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_heuristic_comparison_radar(lang))], width=12)
        ], style={'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_heuristic_metrics_bars(lang))], width=12)
        ], style={'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📋 جدول 4.2: توضیحات استراتژی‌ها", className="text-center"),
                                   style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['primary']}),
                    dbc.CardBody([
                        dbc.Table([
                            html.Thead(html.Tr([
                                html.Th("استراتژی", style={'color': COLORS['primary']}),
                                html.Th("توضیحات", style={'color': COLORS['primary']}),
                                html.Th("پاداش", style={'color': COLORS['primary']})
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td("Conservative", className="text-success"),
                                    html.Td(HEURISTIC_STRATEGIES['Conservative']['description']),
                                    html.Td(f"+{HEURISTIC_STRATEGIES['Conservative']['reward']:.2f}", className="text-warning")
                                ]),
                                html.Tr([
                                    html.Td("Balanced", className="text-primary"),
                                    html.Td(HEURISTIC_STRATEGIES['Balanced']['description']),
                                    html.Td(f"+{HEURISTIC_STRATEGIES['Balanced']['reward']:.2f}", className="text-warning")
                                ]),
                                html.Tr([
                                    html.Td("Adaptive", className="text-info"),
                                    html.Td(HEURISTIC_STRATEGIES['Adaptive']['description']),
                                    html.Td(f"+{HEURISTIC_STRATEGIES['Adaptive']['reward']:.2f}", className="text-warning")
                                ]),
                                html.Tr([
                                    html.Td("Greedy", className="text-danger"),
                                    html.Td(HEURISTIC_STRATEGIES['Greedy']['description']),
                                    html.Td(f"+{HEURISTIC_STRATEGIES['Greedy']['reward']:.2f}", className="text-warning")
                                ])
                            ])
                        ], bordered=True, dark=True, hover=True, responsive=True, striped=True)
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=12)
        ])
    ], fluid=True)


def render_ablation_tab(lang, t):
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_ablation_comparison_chart(lang))], width=12)
        ], style={'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 جدول 4.3: نتایج آماری Ablation", className="text-center"),
                                   style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['primary']}),
                    dbc.CardBody([
                        dbc.Table([
                            html.Thead(html.Tr([
                                html.Th("Variant", style={'color': COLORS['primary']}),
                                html.Th("Best Reward", style={'color': COLORS['primary']}),
                                html.Th("Final Avg", style={'color': COLORS['primary']}),
                                html.Th("Cohen's d", style={'color': COLORS['primary']}),
                                html.Th("p-value", style={'color': COLORS['primary']}),
                                html.Th("معناداری", style={'color': COLORS['primary']})
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td(variant, className="text-success" if variant == "Full Model" else "text-secondary"),
                                    html.Td(f"{data['reward']:.2f}", className="text-warning"),
                                    html.Td(f"{data['final_avg']:.2f}"),
                                    html.Td(f"{data['cohens_d']:.4f}"),
                                    html.Td(f"{data['p_value']:.2e}" if data['p_value'] < 1 else "—"),
                                    html.Td(data['significance'], className="text-info")
                                ]) for variant, data in ABLATION_RESULTS.items()
                            ])
                        ], bordered=True, dark=True, hover=True, responsive=True, striped=True)
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=12)
        ], style={'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("🔬 نتیجه‌گیری Ablation Study", className="alert-heading"),
                    html.Hr(),
                    html.P("⭐ No GAT: افت 27% - اهمیت مدل‌سازی گراف"),
                    html.P("⭐⭐ Decentralized: افت 50% - اهمیت CTDE برای همگرایی"),
                    html.P("⭐⭐⭐ Simpler Arch: افت 65% - ظرفیت شبکه برای یادگیری ضروری است"),
                    html.P("✅ No Temporal: بدون تأثیر منفی - محیط Markovian است", className="text-success")
                ], color="info", style={'fontFamily': 'Vazirmatn'})
            ], width=12)
        ])
    ], fluid=True)


def render_baseline_tab(lang, t):
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_baseline_comparison_chart(lang))], width=12)
        ], style={'marginBottom': '30px'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("📈 برتری MADDPG نسبت به Baseline‌ها", className="alert-heading"),
                    html.Hr(),
                    html.P("🔋 مصرف باتری: 57% کمتر از Random, 54% کمتر از Always Local"),
                    html.P("⏱️ تأخیر: 56% کمتر از Random, 39% کمتر از Always Local"),
                    html.P("✅ نرخ موفقیت: 110% بهتر از Random, 51% بهتر از Always Local"),
                    html.P("🎯 پاداش کل: +130.53 (بهترین عملکرد)", className="text-warning font-weight-bold")
                ], color="success", style={'fontFamily': 'Vazirmatn'})
            ], width=12)
        ])
    ], fluid=True)


def render_complexity_tab(lang, t):
    """🆕 تب تحلیل پیچیدگی"""
    return dbc.Container([
        # Header Section
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H4("🎚️ تحلیل تأثیر پیچیدگی محیط بر عملکرد مدل", className="alert-heading text-center"),
                    html.Hr(),
                    html.P("این بخش نتایج آزمایش مدل MADDPG در سه محیط با سطوح پیچیدگی متفاوت را نمایش می‌دهد (فصل 4، بخش 4.7)",
                           className="text-center")
                ], color="primary", style={'fontFamily': 'Vazirmatn', 'marginBottom': '30px'})
            ], width=12)
        ]),
        
        # Environment Info Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("🟢 Easy", className="text-center text-success"), 
                                   style={'backgroundColor': COLORS['surface_light']}),
                    dbc.CardBody([
                        html.P(f"موانع: {COMPLEXITY_ENVIRONMENTS['Easy']['obstacles']}", className="text-center"),
                        html.P(f"Episodes: {COMPLEXITY_ENVIRONMENTS['Easy']['episodes']}", className="text-center text-secondary")
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("🟠 Medium", className="text-center text-warning"),
                                   style={'backgroundColor': COLORS['surface_light']}),
                    dbc.CardBody([
                        html.P(f"موانع: {COMPLEXITY_ENVIRONMENTS['Medium']['obstacles']}", className="text-center"),
                        html.P(f"Episodes: {COMPLEXITY_ENVIRONMENTS['Medium']['episodes']}", className="text-center text-secondary")
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("🔴 Complex", className="text-center text-danger"),
                                   style={'backgroundColor': COLORS['surface_light']}),
                    dbc.CardBody([
                        html.P(f"موانع: {COMPLEXITY_ENVIRONMENTS['Complex']['obstacles']}", className="text-center"),
                        html.P(f"Episodes: {COMPLEXITY_ENVIRONMENTS['Complex']['episodes']}", className="text-center text-secondary")
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=4),
        ], style={'marginBottom': '30px'}),
        
        # Main Comparison Chart
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_complexity_metrics_comparison(lang))], width=12)
        ], style={'marginBottom': '30px'}),
        
        # Learning Curves
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_learning_curves_complexity(lang))], width=12)
        ], style={'marginBottom': '30px'}),
        
        # Layer Distribution & Training Cost
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_layer_distribution_stacked(lang))], width=6),
            dbc.Col([dcc.Graph(figure=create_training_cost_chart(lang))], width=6),
        ], style={'marginBottom': '30px'}),
        
        # Loss Comparison
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_loss_comparison_complexity(lang))], width=12)
        ], style={'marginBottom': '30px'}),
        
        # Analysis Summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 خلاصه تحلیل نتایج (جداول 4.6 تا 4.13)", className="text-center"),
                                   style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['primary']}),
                    dbc.CardBody([
                        html.Div([
                            html.H6("🔴 تأثیر افزایش پیچیدگی:", className="text-danger"),
                            html.Ul([
                                html.Li(f"مصرف باتری: افزایش 28% (Easy: {COMPLEXITY_PERFORMANCE['Easy']['battery']*100:.2f}% → Complex: {COMPLEXITY_PERFORMANCE['Complex']['battery']*100:.2f}%)"),
                                html.Li(f"تأخیر: افزایش 31.5% (Easy: {COMPLEXITY_PERFORMANCE['Easy']['latency']:.1f}ms → Complex: {COMPLEXITY_PERFORMANCE['Complex']['latency']:.1f}ms)"),
                                html.Li(f"نرخ موفقیت: کاهش 4.2% (Easy: {COMPLEXITY_PERFORMANCE['Easy']['success_rate']:.1f}% → Complex: {COMPLEXITY_PERFORMANCE['Complex']['success_rate']:.1f}%)"),
                            ]),
                            html.Hr(style={'borderColor': COLORS['surface_light']}),
                            
                            html.H6("📈 تأثیر بر یادگیری:", className="text-warning"),
                            html.Ul([
                                html.Li(f"Episode همگرایی: افزایش 80% (Easy: {COMPLEXITY_PERFORMANCE['Easy']['convergence_episode']} → Complex: {COMPLEXITY_PERFORMANCE['Complex']['convergence_episode']})"),
                                html.Li(f"پاداش نهایی: کاهش شدید (Easy: +{COMPLEXITY_PERFORMANCE['Easy']['final_reward']:.2f} → Complex: {COMPLEXITY_PERFORMANCE['Complex']['final_reward']:.2f})"),
                                html.Li(f"Actor Loss نهایی: افزایش 45% (Easy: {COMPLEXITY_PERFORMANCE['Easy']['actor_loss']:.2f} → Complex: {COMPLEXITY_PERFORMANCE['Complex']['actor_loss']:.2f})"),
                            ]),
                            html.Hr(style={'borderColor': COLORS['surface_light']}),
                            
                            html.H6("🌐 تغییر استراتژی انتخاب لایه:", className="text-info"),
                            html.Ul([
                                html.Li(f"Easy: بیشترین استفاده از Terrestrial Edge ({LAYER_DISTRIBUTION_COMPLEXITY['Easy']['terrestrial_edge']:.1f}%)"),
                                html.Li(f"Complex: تغییر به Aerial Edge ({LAYER_DISTRIBUTION_COMPLEXITY['Complex']['aerial_edge']:.1f}%) برای اجتناب از موانع"),
                                html.Li(f"کاهش استفاده از Cloud (Easy: {LAYER_DISTRIBUTION_COMPLEXITY['Easy']['cloud']:.1f}% → Complex: {LAYER_DISTRIBUTION_COMPLEXITY['Complex']['cloud']:.1f}%)"),
                            ]),
                            html.Hr(style={'borderColor': COLORS['surface_light']}),
                            
                            html.H6("💰 هزینه محاسباتی:", className="text-success"),
                            html.Ul([
                                html.Li(f"زمان آموزش: افزایش 56% (Easy: {COMPLEXITY_PERFORMANCE['Easy']['training_hours']}h → Complex: {COMPLEXITY_PERFORMANCE['Complex']['training_hours']}h)"),
                                html.Li("سخت‌افزار: GPU NVIDIA GTX 1660 Ti (6GB VRAM)"),
                            ]),
                            html.Hr(style={'borderColor': COLORS['surface_light']}),
                            
                            html.H6("✅ نتیجه‌گیری:", className="text-primary font-weight-bold"),
                            html.P("مدل MADDPG با وجود افزایش پیچیدگی، قادر به یادگیری سیاست‌های تطبیقی است و به صورت هوشمندانه استراتژی انتخاب لایه را تغییر می‌دهد. "
                                   "با این حال، افزایش پیچیدگی محیط منجر به افزایش قابل توجه هزینه محاسباتی و کاهش عملکرد می‌شود.",
                                   className="text-warning")
                        ], style={'fontFamily': 'Vazirmatn', 'color': COLORS['text']})
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=12)
        ])
    ], fluid=True)


# ========================================
# Layout اصلی
# ========================================

app.layout = dbc.Container([
    dcc.Store(id='lang-store', data='fa'),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("🚁 SkyMind Dashboard v3.1", 
                        style={'color': COLORS['primary'], 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'marginTop': '20px'}),
                html.H4("سیستم هوشمند تخلیه وظایف با MADDPG", 
                        style={'color': COLORS['text_secondary'], 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'marginBottom': '10px'}),
                html.P("📘 فصل 3: طراحی | 📊 فصل 4: نتایج و ارزیابی",
                       style={'color': COLORS['text_secondary'], 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
            ])
        ], width=10),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button("FA", id="btn-fa", color="primary", size="sm", outline=True),
                dbc.Button("EN", id="btn-en", color="secondary", size="sm", outline=True),
            ], style={'marginTop': '30px'})
        ], width=2, className="text-right")
    ], style={'marginBottom': '20px'}),
    
    # Tabs
    dbc.Tabs(id='main-tabs', active_tab='tab-overview', children=[
        dbc.Tab(label='📊 نمای کلی', tab_id='tab-overview', tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label='📈 نتایج آموزش', tab_id='tab-training', tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label='🌐 تحلیل لایه‌ای', tab_id='tab-layer', tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label='⚡ استراتژی‌های Heuristic', tab_id='tab-heuristics', tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label='🔬 Ablation Study', tab_id='tab-ablation', tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label='📉 مقایسه Baseline', tab_id='tab-baseline', tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label='🎚️ تحلیل پیچیدگی', tab_id='tab-complexity', tab_style={'fontFamily': 'Vazirmatn'}),  # 🆕
    ], style={'marginBottom': '30px', 'fontFamily': 'Vazirmatn'}),
    
    # Content
    html.Div(id='tab-content'),
    
    # Footer
    html.Hr(style={'borderColor': COLORS['surface_light'], 'marginTop': '50px'}),
    html.P("© 2025 SkyMind Project | MADDPG-based Task Offloading System", 
           style={'textAlign': 'center', 'color': COLORS['text_secondary'], 'fontFamily': 'Vazirmatn', 'marginBottom': '30px'})
    
], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'paddingTop': '20px'})


# ========================================
# Callbacks
# ========================================

@app.callback(
    Output('lang-store', 'data'),
    [Input('btn-fa', 'n_clicks'), Input('btn-en', 'n_clicks')]
)
def update_language(fa_clicks, en_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'fa'
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    return 'fa' if button_id == 'btn-fa' else 'en'


@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'active_tab'), Input('lang-store', 'data')]
)
def render_tab_content(active_tab, lang):
    t = TRANSLATIONS[lang]
    
    if active_tab == 'tab-overview':
        return render_overview_tab(lang, t)
    elif active_tab == 'tab-training':
        return render_training_tab(lang, t)
    elif active_tab == 'tab-layer':
        return render_layer_tab(lang, t)
    elif active_tab == 'tab-heuristics':
        return render_heuristics_tab(lang, t)
    elif active_tab == 'tab-ablation':
        return render_ablation_tab(lang, t)
    elif active_tab == 'tab-baseline':
        return render_baseline_tab(lang, t)
    elif active_tab == 'tab-complexity':  # 🆕
        return render_complexity_tab(lang, t)
    
    return html.Div()


# ========================================
# اجرا
# ========================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
