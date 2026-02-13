import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ========================================
# تنظیمات رنگ‌بندی (Dark Theme)
# ========================================
COLORS = {
    'background': '#0a0e27',
    'surface': '#141b2d',
    'surface_light': '#1f2940',
    'primary': '#00d9ff',
    'secondary': '#7c3aed',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'text': '#e5e7eb',
    'text_secondary': '#9ca3af'
}

# ========================================
# داده‌های شبیه‌سازی شده
# ========================================

# منحنی یادگیری (1000 Episode)
np.random.seed(42)
episodes = np.arange(0, 1000, 1)
training_rewards = -500 + 530 * (1 - np.exp(-episodes / 200)) + np.random.normal(0, 30, len(episodes))
training_rewards = np.clip(training_rewards, -600, 100)

# Loss curves
actor_loss = 10 * np.exp(-episodes / 150) + np.random.normal(0, 0.5, len(episodes))
critic_loss = 50 * np.exp(-episodes / 120) + np.random.normal(0, 2, len(episodes))

# داده‌های Heuristic Strategy (جدول 4.2)
HEURISTIC_STRATEGIES = {
    'Conservative': {'battery': 119.3, 'latency': 58.2, 'overload': 4.12, 'success': 89.5},
    'Balanced': {'battery': 95.7, 'latency': 54.2, 'overload': 3.82, 'success': 95.0},
    'Adaptive': {'battery': 102.4, 'latency': 51.3, 'overload': 4.65, 'success': 92.3},
    'Greedy': {'battery': 87.6, 'latency': 49.1, 'overload': 6.28, 'success': 78.4}
}

# جدول 4.6: آمار لایه‌ها
LAYER_STATS = {
    'local': {'count': 223, 'percent': 22.3, 'latency': 8.2, 'energy': 12.4},
    'terrestrial_edge': {'count': 458, 'percent': 45.8, 'latency': 24.5, 'energy': 45.3},
    'aerial_edge': {'count': 254, 'percent': 25.4, 'latency': 52.1, 'energy': 78.6},
    'cloud': {'count': 65, 'percent': 6.5, 'latency': 128.7, 'energy': 156.2},
    'reject': {'count': 12, 'percent': 1.2, 'latency': 0, 'energy': 0}
}

# جدول 4.2: مقایسه Ablation (با Actor/Critic Loss و Training Time)
ABLATION_RESULTS = {
    'Full Model': {
        'reward': 130.53,
        'final_avg': -27.41,
        'success': 95.0,
        'actor_loss': 2.14,
        'critic_loss': 1.82,
        'training_time': 3.2,
        'cohens_d': 0.0,
        'p_value': 1.0,
        'significance': '—'
    },
    'No GAT': {
        'reward': -20.24,
        'final_avg': -87.89,
        'success': 47.89,
        'actor_loss': 8.57,
        'critic_loss': 12.3,
        'training_time': 2.8,
        'cohens_d': 0.3774,
        'p_value': 8.57e-3,
        'significance': '⭐'
    },
    'No Temporal': {
        'reward': -26.63,
        'final_avg': -20.90,
        'success': 20.99,
        'actor_loss': 5.94,
        'critic_loss': 8.76,
        'training_time': 2.9,
        'cohens_d': -0.0758,
        'p_value': 5.94e-1,
        'significance': '—'
    },
    'Decentralized': {
        'reward': -85.81,
        'final_avg': -110.86,
        'success': -110.86,
        'actor_loss': 6.52,
        'critic_loss': 15.4,
        'training_time': 3.1,
        'cohens_d': 0.4923,
        'p_value': 6.52e-4,
        'significance': '⭐⭐'
    },
    'Simpler Arch': {
        'reward': -82.69,
        'final_avg': -438.14,
        'success': -438.14,
        'actor_loss': 17.2,
        'critic_loss': 28.6,
        'training_time': 1.5,
        'cohens_d': 1.1250,
        'p_value': 1.72e-13,
        'significance': '⭐⭐⭐'
    }
}

# Baseline Methods (با Energy)
BASELINE_METHODS = {
    'MADDPG (Ours)': {
        'reward': 130.53,
        'latency': 54.23,
        'energy': 45.3,
        'success_rate': 95.0
    },
    'Random': {
        'reward': -245.32,
        'latency': 125.67,
        'energy': 178.6,
        'success_rate': 45.2
    },
    'Conservative': {
        'reward': -89.23,
        'latency': 89.34,
        'energy': 119.3,
        'success_rate': 62.8
    },
    'Balanced': {
        'reward': 45.67,
        'latency': 78.56,
        'energy': 95.7,
        'success_rate': 78.5
    },
    'Adaptive': {
        'reward': 12.89,
        'latency': 92.14,
        'energy': 102.4,
        'success_rate': 71.3
    },
    'Greedy': {
        'reward': 78.45,
        'latency': 68.92,
        'energy': 87.6,
        'success_rate': 82.4
    }
}

# ========================================
# توابع نمودارسازی - بخش Overview
# ========================================

def create_gauge_chart(value, title, max_val, color, delta=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if delta else "gauge+number",
        value=value,
        delta={'reference': delta} if delta else None,
        title={'text': title, 'font': {'size': 16, 'color': COLORS['text']}},
        gauge={
            'axis': {'range': [None, max_val], 'tickcolor': COLORS['text']},
            'bar': {'color': color},
            'bgcolor': COLORS['surface_light'],
            'borderwidth': 2,
            'bordercolor': color,
            'steps': [
                {'range': [0, max_val * 0.33], 'color': COLORS['surface']},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': COLORS['surface_light']},
                {'range': [max_val * 0.66, max_val], 'color': COLORS['background']}
            ],
        },
        number={'font': {'size': 32, 'color': color}}
    ))
    fig.update_layout(
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=250,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def create_state_distribution_pie():
    labels = ['State (402)', 'External (20)', 'Internal (20)', 'Task (40)']
    values = [402, 20, 20, 40]
    colors = [COLORS['primary'], COLORS['success'], COLORS['warning'], COLORS['danger']]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color=COLORS['background'], width=2)),
        textinfo='label+percent',
        textfont=dict(size=14, color=COLORS['text'])
    )])

    fig.update_layout(
        title={'text': 'ساختار فضای حالت (402 بعدی)', 'font': {'size': 18, 'color': COLORS['primary']}},
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
    )
    return fig

def create_action_space_chart():
    categories = ['Offload', 'CPU_alloc', 'BW_alloc', 'UAV_x', 'UAV_y', 'Energy']
    discrete = [5, 0, 0, 0, 0, 0]
    continuous = [0, 30, 50, 20, 20, 100]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Discrete (5 Classes)',
        x=categories,
        y=discrete,
        marker_color=COLORS['success'],
        text=discrete,
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        name='Continuous (6 Params)',
        x=categories,
        y=continuous,
        marker_color=COLORS['primary'],
        text=continuous,
        textposition='outside'
    ))

    fig.update_layout(
        title={'text': 'فضای عمل ترکیبی (Hybrid)', 'font': {'size': 18, 'color': COLORS['primary']}},
        xaxis_title='',
        yaxis_title='',
        barmode='group',
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        xaxis={'gridcolor': COLORS['surface_light']},
        yaxis={'gridcolor': COLORS['surface_light']}
    )
    return fig

# ========================================
# توابع نمودارسازی - بخش Training
# ========================================

def create_training_curve():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=episodes,
        y=training_rewards,
        mode='lines',
        name='منحنی یادگیری (1000 Episode)',
        line=dict(color=COLORS['primary'], width=2),
        fill='tonexty',
        fillcolor='rgba(0, 217, 255, 0.1)'
    ))
    fig.add_hline(y=130.53, line_dash="dash", line_color=COLORS['success'],
                  annotation_text=f"Best: {training_rewards.max():.2f}",
                  annotation_position="right")

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

def create_loss_curves():
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('🎭 Loss منحنی یادگیری', '📉 Loss منحنی Critic'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )

    fig.add_trace(go.Scatter(
        x=episodes, y=actor_loss,
        mode='lines', name='Actor Loss',
        line=dict(color=COLORS['warning'], width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=episodes, y=critic_loss,
        mode='lines', name='Critic Loss',
        line=dict(color=COLORS['secondary'], width=2)
    ), row=1, col=2)

    fig.update_xaxes(title_text="Episode", gridcolor=COLORS['surface_light'])
    fig.update_yaxes(title_text="Loss", gridcolor=COLORS['surface_light'])

    fig.update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=400,
        showlegend=False
    )
    return fig

# ========================================
# توابع نمودارسازی - بخش Layer Analysis
# ========================================

def create_layer_pie_chart():
    labels = list(LAYER_STATS.keys())
    values = [LAYER_STATS[k]['count'] for k in labels]
    percentages = [LAYER_STATS[k]['percent'] for k in labels]

    labels_display = [f"{l.replace('_', ' ').title()}<br>({p:.1f}%)" for l, p in zip(labels, percentages)]

    colors = [COLORS['primary'], COLORS['success'], COLORS['warning'], COLORS['danger'], COLORS['text_secondary']]

    fig = go.Figure(data=[go.Pie(
        labels=labels_display,
        values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color=COLORS['background'], width=2)),
        textinfo='label+value',
        textfont=dict(size=13, color='white')
    )])

    fig.update_layout(
        title={'text': '📊 توزیع انتخاب لایه (جدول 4.1)', 'font': {'size': 18, 'color': COLORS['primary']}},
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=450,
        showlegend=False
    )
    return fig

# ========================================
# توابع نمودارسازی - بخش Heuristic
# ========================================

def create_heuristic_radar():
    categories = ['Battery', 'Latency', 'Overload', 'Success']
    strategies = list(HEURISTIC_STRATEGIES.keys())

    fig = go.Figure()

    colors_radar = [COLORS['success'], COLORS['primary'], COLORS['warning'], COLORS['danger']]

    for i, strategy in enumerate(strategies):
        values = [
            HEURISTIC_STRATEGIES[strategy]['battery'],
            HEURISTIC_STRATEGIES[strategy]['latency'],
            HEURISTIC_STRATEGIES[strategy]['overload'],
            HEURISTIC_STRATEGIES[strategy]['success']
        ]
        values += values[:1]  # Close the loop

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=strategy,
            line=dict(color=colors_radar[i], width=2)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 130], gridcolor=COLORS['surface_light']),
            bgcolor=COLORS['background']
        ),
        title={'text': '🎯 مقایسه‌ راداری استراتژی‌ها', 'font': {'size': 18, 'color': COLORS['primary']}},
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500,
        showlegend=True
    )
    return fig

# ========================================
# توابع نمودارسازی - بخش Ablation
# ========================================

def create_ablation_comparison_chart():
    variants = list(ABLATION_RESULTS.keys())
    rewards = [ABLATION_RESULTS[v]['reward'] for v in variants]
    final_avgs = [ABLATION_RESULTS[v]['final_avg'] for v in variants]

    colors_bars = [COLORS['success'] if r > 0 else COLORS['danger'] for r in rewards]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('🏆 Best Reward', '📊 Final Avg (100 Last)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )

    fig.add_trace(go.Bar(
        x=variants, y=rewards,
        marker_color=colors_bars,
        text=[f"{r:.2f}" for r in rewards],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=variants, y=final_avgs,
        marker_color=COLORS['secondary'],
        text=[f"{r:.2f}" for r in final_avgs],
        textposition='outside',
        showlegend=False
    ), row=1, col=2)

    fig.add_hline(y=130.53, line_dash="dash", line_color=COLORS['success'],
                  annotation_text="Full Model: 130.53", annotation_position="right", row=1, col=1)

    fig.update_xaxes(gridcolor=COLORS['surface_light'])
    fig.update_yaxes(gridcolor=COLORS['surface_light'])

    fig.update_layout(
        title={'text': '🔬 مقایسه Ablation Study (جدول 4.2)', 'font': {'size': 20, 'color': COLORS['primary']}},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500
    )
    return fig

# ========================================
# توابع نمودارسازی - بخش Baseline
# ========================================

def create_baseline_comparison_chart():
    methods = list(BASELINE_METHODS.keys())

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('⏱️ تأخیر (ms)', '⚡ انرژی (mJ)', '✅ نرخ موفقیت (%)', '🎯 پاداش'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]]
    )

    # Latency
    latency_vals = [BASELINE_METHODS[m]['latency'] for m in methods]
    fig.add_trace(go.Bar(x=methods, y=latency_vals, marker_color=COLORS['primary'],
                         text=[f"{v:.1f}" for v in latency_vals], textposition='outside', showlegend=False),
                  row=1, col=1)

    # Energy
    energy_vals = [BASELINE_METHODS[m]['energy'] for m in methods]
    fig.add_trace(go.Bar(x=methods, y=energy_vals, marker_color=COLORS['danger'],
                         text=[f"{v:.1f}" for v in energy_vals], textposition='outside', showlegend=False),
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
        title={'text': 'مقایسه مصرف انرژی (mJ)', 'font': {'size': 18, 'color': COLORS['primary']}},
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
        title={'text': 'تحلیل اجزای تأخیر (ms)', 'font': {'size': 18, 'color': COLORS['primary']}},
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
        title={'text': 'تأثیر پیچیدگی محیط', 'font': {'size': 18, 'color': COLORS['primary']}},
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
        title={'text': 'تغییر استراتژی انتخاب لایه (%)', 'font': {'size': 18, 'color': COLORS['primary']}},
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
        title={'text': 'مقایسه Ablation Study', 'font': {'size': 18, 'color': COLORS['primary']}},
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
        title={'text': 'عملکرد در برابر اهداف تحقیق', 'font': {'size': 18, 'color': COLORS['primary']}},
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500
    )
    return fig


# ========================================
# توابع Render برای هر تب
# ========================================

def render_overview_tab():
    return dbc.Container([
        # ردیف اول: گیج‌ها
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_gauge_chart(95, 'نرخ موفقیت', 100, COLORS['success'], 15))], width=4),
            dbc.Col([dcc.Graph(figure=create_gauge_chart(130.5, 'بهترین پاداش', 150, COLORS['primary'], 10.5))], width=4),
            dbc.Col([dcc.Graph(figure=create_gauge_chart(842, 'همگرایی Episode', 1000, COLORS['warning'], 42))], width=4),
        ], style={'marginBottom': '30px'}),

        # ردیف دوم: نمودارهای State و Action
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_state_distribution_pie())], width=6),
            dbc.Col([dcc.Graph(figure=create_action_space_chart())], width=6),
        ])
    ], fluid=True)

def render_training_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_training_curve())], width=12)
        ], style={'marginBottom': '30px'}),

        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_loss_curves())], width=12)
        ])
    ], fluid=True)

def render_layer_analysis_tab():
    return dbc.Container([
        # نمودار Pie
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_layer_pie_chart())], width=12)
        ], style={'marginBottom': '30px'}),

        # 🆕 جدول 4.6: آمار تفصیلی لایه‌ها
        dbc.Row([
            dbc.Col([
                html.H4("📋 جدول 4.6: آمار تفصیلی عملکرد لایه‌ها", 
                        style={'color': COLORS['primary'], 'fontFamily': 'Vazirmatn', 'marginBottom': '20px'}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th('لایه', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('تعداد', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('درصد (%)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('تأخیر (ms)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('انرژی (mJ)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td('Local', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('223', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('22.3%', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('8.2', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('12.4', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                        ]),
                        html.Tr([
                            html.Td('Terrestrial Edge', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('458', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('45.8%', style={'backgroundColor': COLORS['background'], 'color': COLORS['primary'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('24.5', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('45.3', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                        html.Tr([
                            html.Td('Aerial Edge', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('254', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('25.4%', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('52.1', style={'backgroundColor': COLORS['surface'], 'color': COLORS['warning'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('78.6', style={'backgroundColor': COLORS['surface'], 'color': COLORS['warning'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                        html.Tr([
                            html.Td('Cloud', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('65', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('6.5%', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('128.7', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('156.2', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                        ]),
                        html.Tr([
                            html.Td('Reject', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_secondary'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('12', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_secondary'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('1.2%', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_secondary'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('—', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_secondary'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('—', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_secondary'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': f'1px solid {COLORS["surface_light"]}'})
            ], width=12)
        ])
    ], fluid=True)
def render_heuristics_tab():
    return dbc.Container([
        # نمودار Radar
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_heuristic_radar())], width=12)
        ], style={'marginBottom': '30px'}),

        # 🆕 جدول 4.2: نتایج کمی استراتژی‌های Heuristic
        dbc.Row([
            dbc.Col([
                html.H4("📋 جدول 4.2: مقایسه استراتژی‌های Heuristic", 
                        style={'color': COLORS['primary'], 'fontFamily': 'Vazirmatn', 'marginBottom': '20px'}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th('استراتژی', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('Battery (%)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('Latency (ms)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('Overload (%)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('Success Rate (%)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td('Conservative', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('119.3', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('58.2', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('4.12', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('89.5', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                        ]),
                        html.Tr([
                            html.Td('Balanced', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('95.7', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('54.2', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('3.82', style={'backgroundColor': COLORS['background'], 'color': COLORS['primary'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('95.0', style={'backgroundColor': COLORS['background'], 'color': COLORS['primary'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                        ]),
                        html.Tr([
                            html.Td('Adaptive', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('102.4', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('51.3', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('4.65', style={'backgroundColor': COLORS['surface'], 'color': COLORS['warning'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('92.3', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                        html.Tr([
                            html.Td('Greedy', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('87.6', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('49.1', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('6.28', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('78.4', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': f'1px solid {COLORS["surface_light"]}'})
            ], width=12)
        ])
    ], fluid=True)


def render_ablation_tab():
    return dbc.Container([
        # نمودار مقایسه
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_ablation_comparison_chart())], width=12)
        ], style={'marginBottom': '30px'}),

        # 🆕 جدول 4.2 کامل: شامل Actor Loss, Critic Loss, و زمان آموزش
        dbc.Row([
            dbc.Col([
                html.H4("📋 جدول 4.2: نتایج کامل Ablation Study", 
                        style={'color': COLORS['primary'], 'fontFamily': 'Vazirmatn', 'marginBottom': '20px'}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th('Variant', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '13px'}),
                        html.Th('Best Reward', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '13px'}),
                        html.Th('Final Avg', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '13px'}),
                        html.Th('Actor Loss', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '13px'}),
                        html.Th('Critic Loss', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '13px'}),
                        html.Th('Training (h)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '13px'}),
                        html.Th("Cohen's d", style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '13px'}),
                        html.Th('p-value', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '13px'}),
                        html.Th('Sig.', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '13px'}),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td('Full Model', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold', 'fontSize': '12px'}),
                            html.Td('130.53', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold', 'fontSize': '12px'}),
                            html.Td('-27.41', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('2.14', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('1.82', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('3.2', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('0.0', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_secondary'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('1.0', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_secondary'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('—', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_secondary'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                        ]),
                        html.Tr([
                            html.Td('No GAT', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('-20.24', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('-87.89', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('8.57', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('12.3', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('2.8', style={'backgroundColor': COLORS['background'], 'color': COLORS['success'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('0.3774', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('8.57e-3', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('⭐', style={'backgroundColor': COLORS['background'], 'color': COLORS['warning'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '14px'}),
                        ]),
                        html.Tr([
                            html.Td('No Temporal', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('-26.63', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('-20.90', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('5.94', style={'backgroundColor': COLORS['surface'], 'color': COLORS['warning'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('8.76', style={'backgroundColor': COLORS['surface'], 'color': COLORS['warning'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('2.9', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('-0.0758', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('5.94e-1', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('—', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text_secondary'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                        ]),
                        html.Tr([
                            html.Td('Decentralized', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('-85.81', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold', 'fontSize': '12px'}),
                            html.Td('-110.86', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('6.52', style={'backgroundColor': COLORS['background'], 'color': COLORS['warning'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('15.4', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold', 'fontSize': '12px'}),
                            html.Td('3.1', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('0.4923', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('6.52e-4', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('⭐⭐', style={'backgroundColor': COLORS['background'], 'color': COLORS['warning'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '14px'}),
                        ]),
                        html.Tr([
                            html.Td('Simpler Arch', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('-82.69', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('-438.14', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold', 'fontSize': '12px'}),
                            html.Td('17.2', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold', 'fontSize': '12px'}),
                            html.Td('28.6', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold', 'fontSize': '12px'}),
                            html.Td('1.5', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold', 'fontSize': '12px'}),
                            html.Td('1.1250', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('1.72e-13', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '12px'}),
                            html.Td('⭐⭐⭐', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '8px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontSize': '14px'}),
                        ]),
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': f'1px solid {COLORS["surface_light"]}'})
            ], width=12)
        ])
    ], fluid=True)


def render_baseline_tab():
    return dbc.Container([
        # نمودار مقایسه
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_baseline_comparison_chart())], width=12)
        ], style={'marginBottom': '30px'}),

        # 🆕 جدول کامل Baseline شامل Energy
        dbc.Row([
            dbc.Col([
                html.H4("📋 جدول مقایسه با روش‌های Baseline (شامل انرژی)", 
                        style={'color': COLORS['primary'], 'fontFamily': 'Vazirmatn', 'marginBottom': '20px'}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th('Method', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('Reward', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('Latency (ms)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('Energy (mJ)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('Success Rate (%)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td('MADDPG (Ours)', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('130.53', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('54.23', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('45.3', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('95.0', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                        ]),
                        html.Tr([
                            html.Td('Random', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('-245.32', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('125.67', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('178.6', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('45.2', style={'backgroundColor': COLORS['background'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                        html.Tr([
                            html.Td('Conservative', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('-89.23', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('89.34', style={'backgroundColor': COLORS['surface'], 'color': COLORS['warning'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('119.3', style={'backgroundColor': COLORS['surface'], 'color': COLORS['warning'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('62.8', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                        html.Tr([
                            html.Td('Balanced', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('45.67', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('78.56', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('95.7', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('78.5', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                        html.Tr([
                            html.Td('Adaptive', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('12.89', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('92.14', style={'backgroundColor': COLORS['surface'], 'color': COLORS['warning'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('102.4', style={'backgroundColor': COLORS['surface'], 'color': COLORS['warning'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('71.3', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                        html.Tr([
                            html.Td('Greedy', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('78.45', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('68.92', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('87.6', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('82.4', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': f'1px solid {COLORS["surface_light"]}'})
            ], width=12)
        ])
    ], fluid=True)


# ========================================
# بخش Complexity Analysis (تب جدید)
# ========================================

# داده‌های Complexity (از جدول 4.7 و 4.8 فصل 4)
COMPLEXITY_PERFORMANCE = {
    'Easy': {
        'initial_reward': -17.67,
        'final_reward': 12.34,
        'best_reward': 18.52,
        'convergence_episode': 250,
        'actor_loss': 2.87,
        'critic_loss': 0.082,
        'training_hours': 18
    },
    'Medium': {
        'initial_reward': -23.45,
        'final_reward': 3.67,
        'best_reward': 9.23,
        'convergence_episode': 380,
        'actor_loss': 3.42,
        'critic_loss': 0.127,
        'training_hours': 22
    },
    'Complex': {
        'initial_reward': -35.82,
        'final_reward': -8.91,
        'best_reward': -2.14,
        'convergence_episode': 450,
        'actor_loss': 4.15,
        'critic_loss': 0.198,
        'training_hours': 28
    }
}

def create_complexity_learning_curves():
    """منحنی‌های یادگیری سه محیط"""
    episodes = np.arange(0, 4000, 10)
    
    # شبیه‌سازی منحنی‌های یادگیری
    easy_curve = -17 + 35 * (1 - np.exp(-episodes / 150)) + np.random.normal(0, 2, len(episodes))
    medium_curve = -23 + 27 * (1 - np.exp(-episodes / 220)) + np.random.normal(0, 3, len(episodes))
    complex_curve = -35 + 27 * (1 - np.exp(-episodes / 280)) + np.random.normal(0, 4, len(episodes))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=episodes, y=easy_curve,
        mode='lines', name='Easy (No Obstacles)',
        line=dict(color=COLORS['success'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=episodes, y=medium_curve,
        mode='lines', name='Medium (2 Obstacles)',
        line=dict(color=COLORS['warning'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=episodes, y=complex_curve,
        mode='lines', name='Complex (4 Obstacles)',
        line=dict(color=COLORS['danger'], width=2)
    ))
    
    fig.update_layout(
        title={'text': '📈 مقایسه منحنی یادگیری در سطوح مختلف (جدول 4.8)', 'font': {'size': 18, 'color': COLORS['primary']}},
        xaxis={'title': 'Episode', 'gridcolor': COLORS['surface_light']},
        yaxis={'title': 'Average Reward', 'gridcolor': COLORS['surface_light']},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_complexity_metrics_comparison():
    """مقایسه معیارها در محیط‌های مختلف"""
    levels = list(COMPLEXITY_PERFORMANCE.keys())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🎯 پاداش نهایی', '⏱️ Episode همگرایی', '🎭 Actor Loss', '📉 Critic Loss'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Final Rewards
    final_rewards = [COMPLEXITY_PERFORMANCE[l]['final_reward'] for l in levels]
    colors_final = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    fig.add_trace(go.Bar(
        x=levels, y=final_rewards,
        marker_color=colors_final,
        text=[f"{v:.2f}" for v in final_rewards],
        textposition='outside',
        showlegend=False
    ), row=1, col=1)
    
    # Convergence Episodes
    convergence = [COMPLEXITY_PERFORMANCE[l]['convergence_episode'] for l in levels]
    fig.add_trace(go.Bar(
        x=levels, y=convergence,
        marker_color=colors_final,
        text=convergence,
        textposition='outside',
        showlegend=False
    ), row=1, col=2)
    
    # Actor Loss
    actor_loss = [COMPLEXITY_PERFORMANCE[l]['actor_loss'] for l in levels]
    fig.add_trace(go.Bar(
        x=levels, y=actor_loss,
        marker_color=colors_final,
        text=[f"{v:.2f}" for v in actor_loss],
        textposition='outside',
        showlegend=False
    ), row=2, col=1)
    
    # Critic Loss
    critic_loss = [COMPLEXITY_PERFORMANCE[l]['critic_loss'] for l in levels]
    fig.add_trace(go.Bar(
        x=levels, y=critic_loss,
        marker_color=colors_final,
        text=[f"{v:.3f}" for v in critic_loss],
        textposition='outside',
        showlegend=False
    ), row=2, col=2)
    
    fig.update_layout(
        title={'text': '📊 مقایسه معیارهای کلیدی (جدول 4.7)', 'font': {'size': 18, 'color': COLORS['primary']}},
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['surface'],
        font={'color': COLORS['text'], 'family': 'Vazirmatn'},
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(gridcolor=COLORS['surface_light'])
    fig.update_yaxes(gridcolor=COLORS['surface_light'])
    
    return fig


def render_complexity_tab():
    return dbc.Container([
        # منحنی‌های یادگیری
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_complexity_learning_curves())], width=12)
        ], style={'marginBottom': '30px'}),
        
        # مقایسه معیارها
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_complexity_metrics_comparison())], width=12)
        ], style={'marginBottom': '30px'}),
        
        # جدول کامل مقایسه
        dbc.Row([
            dbc.Col([
                html.H4("📋 جدول 4.7 و 4.8: مقایسه کامل سطوح پیچیدگی", 
                        style={'color': COLORS['primary'], 'fontFamily': 'Vazirmatn', 'marginBottom': '20px'}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th('سطح', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('پاداش اولیه', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('پاداش نهایی', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('بهترین پاداش', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('همگرایی (Ep)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('Actor Loss', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('Critic Loss', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        html.Th('زمان آموزش (h)', style={'backgroundColor': COLORS['surface_light'], 'color': COLORS['text'], 'padding': '12px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td('Easy', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('-17.67', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('12.34', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('18.52', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('250', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('2.87', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('0.082', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('18', style={'backgroundColor': COLORS['surface'], 'color': COLORS['success'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                        html.Tr([
                            html.Td('Medium', style={'backgroundColor': COLORS['background'], 'color': COLORS['warning'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('-23.45', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('3.67', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('9.23', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('380', style={'backgroundColor': COLORS['background'], 'color': COLORS['warning'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('3.42', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('0.127', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('22', style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                        ]),
                        html.Tr([
                            html.Td('Complex', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('-35.82', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('-8.91', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('-2.14', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn'}),
                            html.Td('450', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('4.15', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('0.198', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                            html.Td('28', style={'backgroundColor': COLORS['surface'], 'color': COLORS['danger'], 'padding': '10px', 'textAlign': 'center', 'fontFamily': 'Vazirmatn', 'fontWeight': 'bold'}),
                        ]),
                    ])
                ], style={'width': '100%', 'borderCollapse': 'collapse', 'border': f'1px solid {COLORS["surface_light"]}'})
            ], width=12)
        ])
    ], fluid=True)
def render_final_analysis_tab():
    """تب 8: تحلیل نهایی"""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2('تحلیل نهایی (Final Analysis)', 
                        style={'color': COLORS['primary'], 'textAlign': 'center', 'marginBottom': '10px', 'fontFamily': 'Vazirmatn'}),
                html.P('مقایسه جامع نتایج و تحلیل آماری عملکرد سیستم MADDPG',
                       style={'color': COLORS['text_secondary'], 'textAlign': 'center', 'fontSize': '14px', 'fontFamily': 'Vazirmatn'})
            ], width=12)
        ], style={'marginBottom': '30px'}),
        
        # Row 1: مقایسه انرژی و تأخیر
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_final_energy_comparison())], md=6),
            dbc.Col([dcc.Graph(figure=create_delay_components())], md=6),
        ], style={'marginBottom': '20px'}),
        
        # Row 2: تأثیر پیچیدگی و توزیع لایه
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_complexity_impact())], md=6),
            dbc.Col([dcc.Graph(figure=create_layer_distribution_stacked())], md=6),
        ], style={'marginBottom': '20px'}),
        
        # Row 3: Ablation Study
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_ablation_comparison_final())], md=12),
        ], style={'marginBottom': '20px'}),
        
        # Row 4: نمودار رادار نهایی
        dbc.Row([
            dbc.Col([dcc.Graph(figure=create_overall_radar())], md=12),
        ])
        
    ], fluid=True, style={'backgroundColor': COLORS['background'], 'padding': '20px'})


# ========================================
# Layout اصلی اپلیکیشن
# ========================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🚁 SkyMind Dashboard - MADDPG UAV Offloading System",
                    style={'textAlign': 'center', 'color': COLORS['primary'], 
                           'fontFamily': 'Vazirmatn', 'marginTop': '20px', 'marginBottom': '10px'}),
            html.P("تحلیل عملکرد سیستم تخلیه محاسباتی چندعاملی مبتنی بر یادگیری تقویتی عمیق",
                   style={'textAlign': 'center', 'color': COLORS['text_secondary'], 
                          'fontFamily': 'Vazirmatn', 'fontSize': '16px'})
        ])
    ]),
    
    html.Hr(style={'borderColor': COLORS['surface_light']}),
    
    # Tabs
    dbc.Tabs([
        dbc.Tab(label="📊 نمای کلی", tab_id="overview", tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label="📈 فرآیند آموزش", tab_id="training", tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label="🌐 تحلیل لایه‌ها", tab_id="layers", tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label="⚙️ استراتژی‌های Heuristic", tab_id="heuristics", tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label="🔬 Ablation Study", tab_id="ablation", tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label="📉 مقایسه Baseline", tab_id="baseline", tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label="🎯 تحلیل پیچیدگی", tab_id="complexity", tab_style={'fontFamily': 'Vazirmatn'}),
        dbc.Tab(label="🏆 تحلیل نهایی", tab_id="final_analysis", tab_style={'fontFamily': 'Vazirmatn'}),  # ← تب جدید
    ], id="tabs", active_tab="overview"),
    
    html.Div(id="tab-content", style={'marginTop': '30px', 'marginBottom': '50px'})
    
], fluid=True, style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

# ========================================
# Callbacks
# ========================================

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "overview":
        return render_overview_tab()
    elif active_tab == "training":
        return render_training_tab()
    elif active_tab == "layers":
        return render_layer_analysis_tab()
    elif active_tab == "heuristics":
        return render_heuristics_tab()
    elif active_tab == "ablation":
        return render_ablation_tab()
    elif active_tab == "baseline":
        return render_baseline_tab()
    elif active_tab == "complexity":
        return render_complexity_tab()
    elif active_tab == "final_analysis":  
        return render_final_analysis_tab()
    return html.Div("در حال بارگذاری...")


# ========================================
# اجرای برنامه
# ========================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
