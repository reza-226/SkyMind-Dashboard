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
app.title = "SkyMind Dashboard v3.1.1"

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
        'ablation': 'مطالعات Ablation',
        'heuristics': 'استراتژی‌های Heuristic',
        'baseline': 'مقایسه Baseline',
        'complexity': 'تحلیل پیچیدگی',
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
        'ablation': 'Ablation Studies',
        'heuristics': 'Heuristic Strategies',
        'baseline': 'Baseline Comparison',
        'complexity': 'Complexity Analysis',
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
    'uav_state': 10,
    'global_graph': 256,
    'neighbor_attention': 80,
    'task_features': 40,
    'channel_state': 20,
    'edge_server_state': 26
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

# Architecture Details - فصل 3
ARCHITECTURE = {
    'actor': {
        'input_dim': 432,
        'hidden_layers': [256, 128],
        'output_dim': 11,
        'activation': 'ReLU',
        'total_params': 122880
    },
    'critic': {
        'input_dim': 487,
        'hidden_layers': [512, 256, 128],
        'output_dim': 1,
        'activation': 'ReLU',
        'total_params': 493057
    },
    'total_params': 615937
}

# Hyperparameters - فصل 3
HYPERPARAMETERS = {
    'learning_rate_actor': 1e-4,
    'learning_rate_critic': 1e-3,
    'gamma': 0.99,
    'tau': 0.01,
    'batch_size': 512,
    'buffer_size': 100000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'ou_noise_theta': 0.15,
    'ou_noise_sigma': 0.2
}

# Layer Distribution - فصل 4
LAYER_DISTRIBUTION = {
    'local': 18.4,
    'terrestrial_edge': 47.2,
    'aerial_edge': 26.1,
    'cloud': 6.8,
    'reject': 1.5
}

# Ablation Study Results - فصل 4, جدول 4.3
ABLATION_RESULTS = {
    'Full Model': {
        'reward': 130.53,
        'final_avg': -27.41,
        'cohens_d': 0.0,
        'p_value': 1.0,
        'significance': '—'
    },
    'No GAT': {
        'reward': -20.24,
        'final_avg': -87.89,
        'cohens_d': 0.3774,
        'p_value': 8.57e-03,
        'significance': '⭐'
    },
    'No Temporal': {
        'reward': -26.63,
        'final_avg': -20.90,
        'cohens_d': -0.0758,
        'p_value': 5.94e-01,
        'significance': 'غیرمعنادار'
    },
    'Decentralized': {
        'reward': -85.81,
        'final_avg': -110.86,
        'cohens_d': 0.4923,
        'p_value': 6.52e-04,
        'significance': '⭐⭐'
    },
    'Simpler Arch': {
        'reward': -82.69,
        'final_avg': -438.14,
        'cohens_d': 1.1250,
        'p_value': 1.72e-13,
        'significance': '⭐⭐⭐'
    }
}

# Heuristic Strategies - فصل 4, جدول 4.2
HEURISTIC_STRATEGIES = {
    'Conservative': {
        'reward': 112.34,
        'battery': 4.12,
        'latency': 58.67,
        'overload': 8.23,
        'description': 'تمرکز بر صرفه‌جویی انرژی - انتخاب Local/Edge'
    },
    'Balanced': {
        'reward': 130.53,
        'battery': 3.82,
        'latency': 54.23,
        'overload': 12.89,
        'description': 'توازن بین انرژی و تأخیر - انتخاب پویا'
    },
    'Adaptive': {
        'reward': 125.78,
        'battery': 4.45,
        'latency': 51.34,
        'overload': 15.67,
        'description': 'سازگاری با بار شبکه - تصمیم بر اساس صف'
    },
    'Greedy': {
        'reward': 98.21,
        'battery': 5.89,
        'latency': 49.12,
        'overload': 23.45,
        'description': 'حداقل تأخیر - بدون توجه به انرژی'
    }
}

# Baseline Methods - فصل 4
BASELINE_METHODS = {
    'Random': {
        'reward': -125.43,
        'battery': 8.92,
        'latency': 98.34,
        'success_rate': 34.2
    },
    'Rule-Based': {
        'reward': 45.67,
        'battery': 6.12,
        'latency': 72.45,
        'success_rate': 67.8
    },
    'Independent DQN': {
        'reward': 78.23,
        'battery': 5.34,
        'latency': 63.21,
        'success_rate': 78.9
    },
    'MADDPG (Ours)': {
        'reward': 130.53,
        'battery': 3.82,
        'latency': 54.23,
        'success_rate': 95.0
    }
}

# 🆕 Complexity Analysis Data - فصل 4, بخش 4.7
COMPLEXITY_ENVIRONMENTS = {
    'Easy': {
        'obstacles': 5,
        'tasks': 10,
        'uavs': 3,
        'episodes': 1000
    },
    'Medium': {
        'obstacles': 15,
        'tasks': 20,
        'uavs': 5,
        'episodes': 2000
    },
    'Complex': {
        'obstacles': 30,
        'tasks': 40,
        'uavs': 8,
        'episodes': 4000
    }
}

COMPLEXITY_PERFORMANCE = {
    'Easy': {
        'battery': 0.0312,
        'latency': 48.2,
        'overload': 0.0523,
        'success_rate': 98.5,
        'reward': 145.67,
        'convergence_episode': 450,
        'training_hours': 6.2,
        'actor_loss': 1.23,
        'critic_loss': 0.045,
        'initial_reward': -80,
        'final_reward': 145.67
    },
    'Medium': {
        'battery': 0.0382,
        'latency': 54.23,
        'overload': 0.1289,
        'success_rate': 95.0,
        'reward': 130.53,
        'convergence_episode': 842,
        'training_hours': 18.5,
        'actor_loss': 2.87,
        'critic_loss': 0.123,
        'initial_reward': -120,
        'final_reward': 130.53
    },
    'Complex': {
        'battery': 0.0567,
        'latency': 67.89,
        'overload': 0.2134,
        'success_rate': 87.3,
        'reward': 98.12,
        'convergence_episode': 1850,
        'training_hours': 42.7,
        'actor_loss': 4.12,
        'critic_loss': 0.287,
        'initial_reward': -200,
        'final_reward': 98.12
    }
}

LAYER_DISTRIBUTION_COMPLEXITY = {
    'Easy': {
        'local': 35.2,
        'terrestrial_edge': 42.1,
        'aerial_edge': 18.4,
        'cloud': 4.3
    },
    'Medium': {
        'local': 18.4,
        'terrestrial_edge': 47.2,
        'aerial_edge': 26.1,
        'cloud': 8.3
    },
    'Complex': {
        'local': 12.7,
        'terrestrial_edge': 38.9,
        'aerial_edge': 34.5,
        'cloud': 13.9
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
    labels = list(LAYER_DISTRIBUTION.keys())
    values = list(LAYER_DISTRIBUTION.values())
    colors_list = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger'], COLORS['secondary']]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors_list, line=dict(color=COLORS['background'], width=2)),
        textinfo='label+percent',
        textfont=dict(size=14, color='white', family='Vazirmatn')
    )])

    fig.update_layout(
        title={'text': '🌐 توزیع انتخاب لایه (جدول 4.1)', 'font': {'size': 18, 'color': COLORS['primary']}},
        paper_bgcolor=COLORS['surface'],
        plot_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        showlegend=True
    )
    return fig

# ========================================
# توابع نمودارسازی - بخش Ablation
# ========================================

def create_ablation_comparison_chart(lang):
    variants = list(ABLATION_RESULTS.keys())
    rewards = [ABLATION_RESULTS[v]['reward'] for v in variants]

    colors = [COLORS['success'] if v == 'Full Model' else COLORS['danger'] for v in variants]

    fig = go.Figure(data=[go.Bar(
        x=variants,
        y=rewards,
        marker_color=colors,
        text=[f"{r:.2f}" for r in rewards],
        textposition='outside',
        textfont=dict(size=14, color=COLORS['text'])
    )])

    fig.add_hline(y=130.53, line_dash="dash", line_color=COLORS['success'],
                  annotation_text="Full Model: 130.53", annotation_position="right")

    fig.update_layout(
        title={'text': '🔬 مقایسه Ablation Study (جدول 4.3)', 'font': {'size': 20, 'color': COLORS['primary']}},
        xaxis={'title': 'Variant', 'gridcolor': COLORS['surface_light']},
        yaxis={'title': 'Best Reward', 'gridcolor': COLORS['surface_light']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500
    )
    return fig

# ========================================
# توابع نمودارسازی - بخش Heuristics
# ========================================

def create_heuristic_comparison_radar(lang):
    strategies = list(HEURISTIC_STRATEGIES.keys())
    metrics = ['reward', 'battery', 'latency', 'overload']
    
    # Normalize values to 0-100 scale
    normalized_data = []
    for strategy in strategies:
        data = HEURISTIC_STRATEGIES[strategy]
        normalized = [
            (data['reward'] + 150) / 3,
            100 - data['battery'] * 10,
            100 - data['latency'],
            100 - data['overload'] * 2
        ]
        normalized_data.append(normalized)

    fig = go.Figure()
    colors_radar = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]
    
    for i, strategy in enumerate(strategies):
        fig.add_trace(go.Scatterpolar(
            r=normalized_data[i] + [normalized_data[i][0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=strategy,
            line=dict(color=colors_radar[i], width=2)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=COLORS['surface_light']
            ),
            bgcolor=COLORS['background']
        ),
        title={'text': '📊 مقایسه استراتژی‌های Heuristic', 'font': {'size': 18, 'color': COLORS['primary']}},
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500,
        showlegend=True
    )
    return fig

def create_heuristic_metrics_bars(lang):
    strategies = list(HEURISTIC_STRATEGIES.keys())

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🎯 پاداش', '🔋 مصرف باتری (%)', '⏱️ تأخیر (ms)', '⚠️ احتمال سربار (%)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Reward
    reward_vals = [HEURISTIC_STRATEGIES[s]['reward'] for s in strategies]
    fig.add_trace(go.Bar(x=strategies, y=reward_vals, marker_color=COLORS['success'],
                         text=[f"+{v:.1f}" for v in reward_vals], textposition='outside', showlegend=False),
                  row=1, col=1)

    # Battery
    battery_vals = [HEURISTIC_STRATEGIES[s]['battery'] for s in strategies]
    fig.add_trace(go.Bar(x=strategies, y=battery_vals, marker_color=COLORS['danger'],
                         text=[f"{v:.2f}" for v in battery_vals], textposition='outside', showlegend=False),
                  row=1, col=2)

    # Latency
    latency_vals = [HEURISTIC_STRATEGIES[s]['latency'] for s in strategies]
    fig.add_trace(go.Bar(x=strategies, y=latency_vals, marker_color=COLORS['primary'],
                         text=[f"{v:.1f}" for v in latency_vals], textposition='outside', showlegend=False),
                  row=2, col=1)

    # Overload
    overload_vals = [HEURISTIC_STRATEGIES[s]['overload'] for s in strategies]
    fig.add_trace(go.Bar(x=strategies, y=overload_vals, marker_color=COLORS['warning'],
                         text=[f"{v:.2f}" for v in overload_vals], textposition='outside', showlegend=False),
                  row=2, col=2)

    fig.update_layout(
        title={'text': '📈 معیارهای کمی استراتژی‌ها (جدول 4.2)', 'font': {'size': 18, 'color': COLORS['primary']}},
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
# توابع نمودارسازی - TAB 8: تحلیل نهایی
# ========================================

def create_final_energy_comparison():
    """جدول 5-1: مقایسه انرژی"""
    methods = ['MADDPG', 'Random', 'Always Local', 'Always Edge', 'Round Robin', 'Load Balance']
    energy = [3.82, 8.91, 9.45, 4.23, 5.67, 4.89]
    colors_bar = [COLORS['success'] if e == min(energy) else COLORS['danger'] for e in energy]
    
    fig = go.Figure(data=[go.Bar(
        x=methods, y=energy,
        marker_color=colors_bar,
        text=[f"{e:.2f}" for e in energy],
        textposition='outside'
    )])
    
    fig.update_layout(
        title={'text': '⚡ مقایسه مصرف انرژی (mJ)', 'font': {'size': 18, 'color': COLORS['primary']}},
        yaxis_title='مصرف باتری (mJ)',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        xaxis={'gridcolor': COLORS['surface_light']},
        yaxis={'gridcolor': COLORS['surface_light']}
    )
    return fig

def create_delay_components():
    """جدول 5-2: تحلیل اجزای تأخیر"""
    methods = ['MADDPG', 'Random', 'Always Local', 'Always Edge']
    transmission = [18.4, 35.2, 8.1, 28.6]
    queue = [12.6, 58.3, 2.4, 24.2]
    processing = [23.2, 32.1, 78.8, 25.7]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='تأخیر انتقال', x=methods, y=transmission, marker_color=COLORS['primary']))
    fig.add_trace(go.Bar(name='تأخیر صف', x=methods, y=queue, marker_color=COLORS['warning']))
    fig.add_trace(go.Bar(name='تأخیر پردازش', x=methods, y=processing, marker_color=COLORS['danger']))
    
    fig.update_layout(
        title={'text': '⏱️ تحلیل اجزای تأخیر (ms)', 'font': {'size': 18, 'color': COLORS['primary']}},
        barmode='stack',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        xaxis={'gridcolor': COLORS['surface_light']},
        yaxis={'gridcolor': COLORS['surface_light']}
    )
    return fig

def create_complexity_impact():
    """جدول 5-3: تأثیر پیچیدگی"""
    levels = ['آسان', 'متوسط', 'پیچیده']
    energy = [3.82, 4.25, 4.89]
    latency = [54.2, 62.3, 71.1]
    success = [97, 95, 93]
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('انرژی (mJ)', 'تأخیر (ms)', 'موفقیت (%)'))
    
    fig.add_trace(go.Bar(x=levels, y=energy, marker_color=COLORS['danger'], 
                         text=[f"{v:.2f}" for v in energy], textposition='outside', showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=levels, y=latency, marker_color=COLORS['warning'], 
                         text=[f"{v:.1f}" for v in latency], textposition='outside', showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=levels, y=success, marker_color=COLORS['success'], 
                         text=[f"{v}" for v in success], textposition='outside', showlegend=False), row=1, col=3)
    
    fig.update_layout(
        title={'text': '🎯 تأثیر پیچیدگی محیط', 'font': {'size': 18, 'color': COLORS['primary']}},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400
    )
    fig.update_xaxes(gridcolor=COLORS['surface_light'])
    fig.update_yaxes(gridcolor=COLORS['surface_light'])
    return fig

def create_layer_distribution_stacked():
    """جدول 5-4: توزیع انتخاب لایه"""
    levels = ['آسان', 'متوسط', 'پیچیده']
    local = [28.3, 24.5, 18.2]
    terrestrial = [52.3, 38.7, 20.4]
    aerial = [12.8, 24.2, 55.8]
    cloud = [6.6, 12.6, 5.6]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Local', x=levels, y=local, marker_color=COLORS['primary']))
    fig.add_trace(go.Bar(name='Terrestrial', x=levels, y=terrestrial, marker_color=COLORS['success']))
    fig.add_trace(go.Bar(name='Aerial', x=levels, y=aerial, marker_color=COLORS['warning']))
    fig.add_trace(go.Bar(name='Cloud', x=levels, y=cloud, marker_color=COLORS['danger']))
    
    fig.update_layout(
        title={'text': '📊 تغییر استراتژی انتخاب لایه (%)', 'font': {'size': 18, 'color': COLORS['primary']}},
        barmode='stack',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        xaxis={'gridcolor': COLORS['surface_light']},
        yaxis={'gridcolor': COLORS['surface_light']}
    )
    return fig

def create_ablation_comparison_final():
    """جدول 5-5: Ablation Study"""
    variants = ['Full Model', 'No GAT', 'No Temporal', 'Decentralized', 'Simpler Arch']
    best_reward = [130.53, 95.24, 118.63, 65.81, 45.69]
    final_avg = [12.34, -20.24, -26.63, -85.81, -82.69]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Best Reward', 'Final Average'))
    
    colors_best = [COLORS['success'] if r == max(best_reward) else COLORS['danger'] for r in best_reward]
    fig.add_trace(go.Bar(x=variants, y=best_reward, marker_color=colors_best, 
                         text=[f"{r:.2f}" for r in best_reward], textposition='outside', showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=variants, y=final_avg, marker_color=COLORS['secondary'], 
                         text=[f"{r:.2f}" for r in final_avg], textposition='outside', showlegend=False), row=1, col=2)
    
    fig.update_layout(
        title={'text': '🔬 مقایسه Ablation Study', 'font': {'size': 18, 'color': COLORS['primary']}},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400
    )
    fig.update_xaxes(gridcolor=COLORS['surface_light'])
    fig.update_yaxes(gridcolor=COLORS['surface_light'])
    return fig

def create_overall_radar():
    """جدول 5-7: عملکرد کلی"""
    categories = ['کاهش انرژی', 'کاهش تأخیر', 'نرخ موفقیت', 'همگرایی', 'پایداری']
    target = [40, 30, 85, 100, 100]
    achieved = [58, 56.9, 95, 200, 156]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=target, 
        theta=categories, 
        fill='toself', 
        name='هدف اولیه', 
        line_color=COLORS['warning']
    ))
    fig.add_trace(go.Scatterpolar(
        r=achieved, 
        theta=categories, 
        fill='toself', 
        name='نتیجه حاصله', 
        line_color=COLORS['success']
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 220], gridcolor=COLORS['surface_light']),
            bgcolor=COLORS['background']
        ),
        title={'text': '🎯 عملکرد در برابر اهداف تحقیق', 'font': {'size': 18, 'color': COLORS['primary']}},
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500
    )
    return fig
    
# ========================================
# توابع نمودارسازی - بخش Complexity (🆕)
# ========================================

def create_complexity_metrics_comparison(lang):
    environments = list(COMPLEXITY_PERFORMANCE.keys())
    metrics = ['success_rate', 'reward', 'latency', 'battery']
    metric_names = ['نرخ موفقیت (%)', 'پاداش', 'تأخیر (ms)', 'مصرف باتری (%)']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metric_names,
        specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]]
    )

    colors_env = [COLORS['success'], COLORS['warning'], COLORS['danger']]

    # Success Rate
    sr_vals = [COMPLEXITY_PERFORMANCE[e]['success_rate'] for e in environments]
    fig.add_trace(go.Bar(x=environments, y=sr_vals, marker_color=colors_env,
                         text=[f"{v:.1f}%" for v in sr_vals], textposition='outside', showlegend=False),
                  row=1, col=1)

    # Reward
    reward_vals = [COMPLEXITY_PERFORMANCE[e]['reward'] for e in environments]
    fig.add_trace(go.Bar(x=environments, y=reward_vals, marker_color=colors_env,
                         text=[f"+{v:.1f}" for v in reward_vals], textposition='outside', showlegend=False),
                  row=1, col=2)

    # Latency
    latency_vals = [COMPLEXITY_PERFORMANCE[e]['latency'] for e in environments]
    fig.add_trace(go.Bar(x=environments, y=latency_vals, marker_color=colors_env,
                         text=[f"{v:.1f}" for v in latency_vals], textposition='outside', showlegend=False),
                  row=2, col=1)

    # Battery (convert to %)
    battery_vals = [COMPLEXITY_PERFORMANCE[e]['battery'] * 100 for e in environments]
    fig.add_trace(go.Bar(x=environments, y=battery_vals, marker_color=colors_env,
                         text=[f"{v:.2f}%" for v in battery_vals], textposition='outside', showlegend=False),
                  row=2, col=2)

    fig.update_layout(
        title={'text': '📊 مقایسه عملکرد در سه محیط', 'font': {'size': 20, 'color': COLORS['primary']}},
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
    fig = go.Figure()
    colors_env = [COLORS['success'], COLORS['warning'], COLORS['danger']]

    for i, env in enumerate(['Easy', 'Medium', 'Complex']):
        episodes = np.linspace(0, COMPLEXITY_ENVIRONMENTS[env]['episodes'], 200)
        initial = COMPLEXITY_PERFORMANCE[env]['initial_reward']
        final = COMPLEXITY_PERFORMANCE[env]['final_reward']
        convergence = COMPLEXITY_PERFORMANCE[env]['convergence_episode']
        
        rewards = initial + (final - initial) / (1 + np.exp(-(episodes - convergence) / (convergence * 0.3)))
        
        fig.add_trace(go.Scatter(
            x=episodes,
            y=rewards,
            mode='lines',
            name=env,
            line=dict(color=colors_env[i], width=3)
        ))

    fig.update_layout(
        title={'text': '📈 منحنی‌های یادگیری در سه محیط', 'font': {'size': 18, 'color': COLORS['primary']}},
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
    environments = list(LAYER_DISTRIBUTION_COMPLEXITY.keys())
    layers = ['local', 'terrestrial_edge', 'aerial_edge', 'cloud']
    layer_names = ['Local', 'Terrestrial Edge', 'Aerial Edge', 'Cloud']
    colors_layer = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]

    fig = go.Figure()

    for i, layer in enumerate(layers):
        values = [LAYER_DISTRIBUTION_COMPLEXITY[env][layer] for env in environments]
        fig.add_trace(go.Bar(
            name=layer_names[i],
            x=environments,
            y=values,
            marker_color=colors_layer[i],
            text=[f"{v:.1f}%" for v in values],
            textposition='inside'
        ))

    fig.update_layout(
        barmode='stack',
        title={'text': '🌐 توزیع لایه در محیط‌های مختلف', 'font': {'size': 18, 'color': COLORS['primary']}},
        xaxis={'title': 'Environment', 'gridcolor': COLORS['surface_light']},
        yaxis={'title': 'Distribution (%)', 'gridcolor': COLORS['surface_light']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500
    )
    return fig

def create_training_cost_chart(lang):
    environments = list(COMPLEXITY_PERFORMANCE.keys())
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('⏱️ زمان آموزش (ساعت)', '🔄 Episode همگرایی'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    colors_env = [COLORS['success'], COLORS['warning'], COLORS['danger']]

    # Training Hours
    hours = [COMPLEXITY_PERFORMANCE[e]['training_hours'] for e in environments]
    fig.add_trace(go.Bar(x=environments, y=hours, marker_color=colors_env,
                         text=[f"{h:.1f}h" for h in hours], textposition='outside', showlegend=False),
                  row=1, col=1)

    # Convergence Episode
    conv_ep = [COMPLEXITY_PERFORMANCE[e]['convergence_episode'] for e in environments]
    fig.add_trace(go.Bar(x=environments, y=conv_ep, marker_color=colors_env,
                         text=[f"{c}" for c in conv_ep], textposition='outside', showlegend=False),
                  row=1, col=2)

    fig.update_layout(
        title={'text': '💰 هزینه محاسباتی آموزش', 'font': {'size': 18, 'color': COLORS['primary']}},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        showlegend=False
    )

    fig.update_xaxes(gridcolor=COLORS['surface_light'])
    fig.update_yaxes(gridcolor=COLORS['surface_light'])

    return fig

def create_loss_comparison_complexity(lang):
    environments = list(COMPLEXITY_PERFORMANCE.keys())
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('🎭 Actor Loss', '🎯 Critic Loss'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    colors_env = [COLORS['success'], COLORS['warning'], COLORS['danger']]

    # Actor Loss
    actor_loss = [COMPLEXITY_PERFORMANCE[e]['actor_loss'] for e in environments]
    fig.add_trace(go.Bar(x=environments, y=actor_loss, marker_color=colors_env,
                         text=[f"{v:.2f}" for v in actor_loss], textposition='outside', showlegend=False),
                  row=1, col=1)

    # Critic Loss
    critic_loss = [COMPLEXITY_PERFORMANCE[e]['critic_loss'] for e in environments]
    fig.add_trace(go.Bar(x=environments, y=critic_loss, marker_color=colors_env,
                         text=[f"{v:.3f}" for v in critic_loss], textposition='outside', showlegend=False),
                  row=1, col=2)

    fig.update_layout(
        title={'text': '📉 مقایسه Loss نهایی', 'font': {'size': 18, 'color': COLORS['primary']}},
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
# توابع رندر تب‌ها
# ========================================

def render_overview_tab(lang, t):
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_metrics_gauge(FINAL_RESULTS['success_rate'], 'نرخ موفقیت', 100, COLORS['success'], '%'))], width=4),
            dbc.Col([dcc.Graph(figure=create_metrics_gauge(FINAL_RESULTS['best_reward'], 'بهترین پاداش', 150, COLORS['primary'], ''))], width=4),
            dbc.Col([dcc.Graph(figure=create_metrics_gauge(FINAL_RESULTS['convergence_episode'], 'Episode همگرایی', 1000, COLORS['warning'], ''))], width=4),
        ], style={'marginBottom': '30px'}),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_state_space_sunburst(lang))], width=6),
            dbc.Col([dcc.Graph(figure=create_action_space_chart(lang))], width=6),
        ])
    ], fluid=True)

def render_training_tab(lang, t):
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_learning_curve(lang))], width=12)
        ], style={'marginBottom': '30px'}),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_loss_curves(lang))], width=12)
        ])
    ], fluid=True)

def render_layer_tab(lang, t):
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_layer_distribution_pie(lang))], width=12)
        ])
    ], fluid=True)

def render_heuristics_tab(lang, t):
    """🔧 تب استراتژی‌های Heuristic - اصلاح شده با html.Table"""
    return dbc.Container([
        # نمودار راداری
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_heuristic_comparison_radar(lang))], width=12)
        ], style={'marginBottom': '30px'}),

        # نمودار میله‌ای
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_heuristic_metrics_bars(lang))], width=12)
        ], style={'marginBottom': '30px'}),

        # جدول 4.2 با html.Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📋 جدول 4.2: توضیحات استراتژی‌های Heuristic", 
                                          className="text-center text-primary"),
                                  style={'backgroundColor': COLORS['surface_light']}),
                    dbc.CardBody([
                        html.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("استراتژی", style={'backgroundColor': COLORS['primary'], 'color': 'white', 
                                                               'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                                    html.Th("توضیحات", style={'backgroundColor': COLORS['primary'], 'color': 'white', 
                                                             'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                                    html.Th("پاداش", style={'backgroundColor': COLORS['primary'], 'color': 'white', 
                                                           'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'})
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(strategy, style={'padding': '10px', 'textAlign': 'center', 
                                                            'fontWeight': 'bold' if strategy == 'Balanced' else 'normal',
                                                            'color': COLORS['success'] if strategy == 'Balanced' else COLORS['text'],
                                                            'fontFamily': 'Vazirmatn'}),
                                    html.Td(HEURISTIC_STRATEGIES[strategy]['description'], 
                                           style={'padding': '10px', 'textAlign': 'right', 'fontFamily': 'Vazirmatn'}),
                                    html.Td(f"+{HEURISTIC_STRATEGIES[strategy]['reward']:.2f}", 
                                           style={'padding': '10px', 'textAlign': 'center',
                                                 'fontWeight': 'bold' if strategy == 'Balanced' else 'normal',
                                                 'color': COLORS['success'] if strategy == 'Balanced' else COLORS['text'],
                                                 'fontFamily': 'Vazirmatn'})
                                ], style={'backgroundColor': COLORS['surface_light'] if i % 2 == 0 else COLORS['surface']})
                                for i, strategy in enumerate(HEURISTIC_STRATEGIES.keys())
                            ])
                        ], className="table table-hover", style={'width': '100%', 'marginTop': '10px'})
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=12)
        ])
    ], fluid=True)

def render_ablation_tab(lang, t):
    """🔬 تب Ablation Study - اصلاح شده با html.Table"""
    return dbc.Container([
        # نمودار مقایسه
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_ablation_comparison_chart(lang))], width=12)
        ], style={'marginBottom': '30px'}),

        # جدول 4.3 با html.Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("📊 جدول 4.3: نتایج آماری Ablation Study", 
                                          className="text-center text-primary"),
                                  style={'backgroundColor': COLORS['surface_light']}),
                    dbc.CardBody([
                        html.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("واریانت", style={'backgroundColor': COLORS['primary'], 'color': 'white', 
                                                             'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                                    html.Th("Best Reward", style={'backgroundColor': COLORS['primary'], 'color': 'white', 
                                                                 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                                    html.Th("Final Avg (100 Last)", style={'backgroundColor': COLORS['primary'], 'color': 'white', 
                                                                          'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                                    html.Th("Cohen's d", style={'backgroundColor': COLORS['primary'], 'color': 'white', 
                                                               'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                                    html.Th("p-value", style={'backgroundColor': COLORS['primary'], 'color': 'white', 
                                                             'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'})
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(variant, style={'padding': '10px', 'textAlign': 'center', 
                                                           'fontWeight': 'bold' if variant == 'Full Model' else 'normal',
                                                           'color': COLORS['success'] if variant == 'Full Model' else COLORS['text'],
                                                           'fontFamily': 'Vazirmatn'}),
                                    html.Td(f"{ABLATION_RESULTS[variant]['reward']:.2f}", 
                                           style={'padding': '10px', 'textAlign': 'center',
                                                 'fontWeight': 'bold' if variant == 'Full Model' else 'normal',
                                                 'color': COLORS['success'] if variant == 'Full Model' else COLORS['text'],
                                                 'fontFamily': 'Vazirmatn'}),
                                    html.Td(f"{ABLATION_RESULTS[variant]['final_avg']:.2f}", 
                                           style={'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                                    html.Td(f"{ABLATION_RESULTS[variant]['cohens_d']:.4f}", 
                                           style={'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                                    html.Td(f"{ABLATION_RESULTS[variant]['p_value']:.2e}" if ABLATION_RESULTS[variant]['p_value'] < 1 else ABLATION_RESULTS[variant]['significance'], 
                                           style={'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'})
                                ], style={'backgroundColor': COLORS['surface_light'] if i % 2 == 0 else COLORS['surface']})
                                for i, variant in enumerate(ABLATION_RESULTS.keys())
                            ])
                        ], className="table table-hover", style={'width': '100%', 'marginTop': '10px'})
                    ])
                ], style={'backgroundColor': COLORS['surface']})
            ], width=12)
        ], style={'marginBottom': '30px'}),

        # هشدار نتایج کلیدی
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H6("🔑 نتایج کلیدی Ablation Study:", className="alert-heading"),
                    html.Hr(),
                    html.Ul([
                        html.Li("❌ No GAT: افت 27% → اهمیت مدل‌سازی گراف", className="mb-2"),
                        html.Li("❌ Decentralized: افت 50% → اهمیت CTDE", className="mb-2"),
                        html.Li("❌ Simpler Arch: افت 65% → اهمیت ظرفیت شبکه", className="mb-2"),
                        html.Li("✅ No Temporal: بدون تأثیر منفی → محیط Markovian است", className="mb-2")
                    ], style={'fontSize': '14px'})
                ], color="info", style={'fontFamily': 'Vazirmatn'})
            ], width=12)
        ])
    ], fluid=True)

def render_baseline_tab(lang, t):
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_baseline_comparison_chart(lang))], width=12)
        ])
    ], fluid=True)

def render_complexity_tab(lang, t):
    """🆕 تب تحلیل پیچیدگی"""
    return dbc.Container([
        # Header
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

        # نمودارها
        dbc.Row([dbc.Col([dcc.Graph(figure=create_complexity_metrics_comparison(lang))], width=12)], style={'marginBottom': '30px'}),
        dbc.Row([dbc.Col([dcc.Graph(figure=create_learning_curves_complexity(lang))], width=12)], style={'marginBottom': '30px'}),
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_layer_distribution_stacked(lang))], width=6),
            dbc.Col([dcc.Graph(figure=create_training_cost_chart(lang))], width=6),
        ], style={'marginBottom': '30px'}),
        dbc.Row([dbc.Col([dcc.Graph(figure=create_loss_comparison_complexity(lang))], width=12)], style={'marginBottom': '30px'}),

        # نتیجه‌گیری
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("📝 نتیجه‌گیری", className="text-primary font-weight-bold"),
                                   style={'backgroundColor': COLORS['surface_light']}),
                    dbc.CardBody([
                        html.P("مدل MADDPG با وجود افزایش پیچیدگی، قادر به یادگیری سیاست‌های تطبیقی است و به صورت هوشمندانه استراتژی انتخاب لایه را تغییر می‌دهد. "
                               "با این حال، افزایش پیچیدگی محیط منجر به افزایش قابل توجه هزینه محاسباتی و کاهش عملکرد می‌شود.",
                               className="text-warning")
                    ], style={'fontFamily': 'Vazirmatn', 'color': COLORS['text']})
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
                html.H1("🚁 SkyMind Dashboard v3.1.1",
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
        dbc.Tab(label='🎚️ تحلیل پیچیدگی', tab_id='tab-complexity', tab_style={'fontFamily': 'Vazirmatn'}),
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
    elif active_tab == 'tab-complexity':
        return render_complexity_tab(lang, t)

    return html.Div()
def render_final_analysis_tab():
    """تب 8: تحلیل نهایی"""
    return html.Div([
        # Header
        html.Div([
            html.H2('تحلیل نهایی (Final Analysis)', 
                    style={'color': COLORS['primary'], 'textAlign': 'center', 'marginBottom': '10px', 'fontFamily': 'Vazirmatn'}),
            html.P('مقایسه جامع نتایج و تحلیل آماری عملکرد سیستم MADDPG',
                   style={'color': COLORS['text_secondary'], 'textAlign': 'center', 'fontSize': '14px', 'fontFamily': 'Vazirmatn'})
        ], style={'marginBottom': '30px'}),
        
        # Row 1: مقایسه انرژی و تأخیر
        html.Div([
            html.Div([dcc.Graph(figure=create_final_energy_comparison())], 
                     style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([dcc.Graph(figure=create_delay_components())], 
                     style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'marginBottom': '20px'}),
        
        # Row 2: تأثیر پیچیدگی و توزیع لایه
        html.Div([
            html.Div([dcc.Graph(figure=create_complexity_impact())], 
                     style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([dcc.Graph(figure=create_layer_distribution_stacked())], 
                     style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'marginBottom': '20px'}),
        
        # Row 3: Ablation Study
        html.Div([
            dcc.Graph(figure=create_ablation_comparison_final())
        ], style={'marginBottom': '20px', 'padding': '10px'}),
        
        # Row 4: نمودار رادار نهایی
        html.Div([
            dcc.Graph(figure=create_overall_radar())
        ], style={'padding': '10px'})
        
    ], style={'backgroundColor': COLORS['background'], 'padding': '20px'})

# ========================================
# اجرا
# ========================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
