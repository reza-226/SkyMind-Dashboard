# نام فایل: view_training_dashboard.py
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from pathlib import Path

print("📊 Loading Training Results Dashboard...")

# ✅ خواندن داده‌های هر 3 سطح
levels_data = {}
level_paths = {
    'Level 1 (Simple)': 'models/level1_simple/training_history.json',
    'Level 2 (Medium)': 'models/level2_medium/training_history.json',
    'Level 3 (Complex)': 'models/level3_complex/training_history.json'
}

for level_name, path in level_paths.items():
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            levels_data[level_name] = data
            print(f"   ✅ Loaded: {level_name}")
    except FileNotFoundError:
        print(f"   ❌ Not found: {level_name}")
        levels_data[level_name] = None

# ✅ ساخت داشبورد
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# 🎨 رنگ‌های سطوح
colors = {
    'Level 1 (Simple)': '#00ff41',
    'Level 2 (Medium)': '#ffa500', 
    'Level 3 (Complex)': '#ff4444'
}

# 📊 نمودار 1: مقایسه Average Reward
fig_reward = go.Figure()
for level_name, data in levels_data.items():
    if data:
        rewards = data.get('average_rewards', [])
        episodes = list(range(1, len(rewards) + 1))
        fig_reward.add_trace(go.Scatter(
            x=episodes,
            y=rewards,
            mode='lines',
            name=level_name,
            line=dict(width=2, color=colors[level_name])
        ))

fig_reward.update_layout(
    title="📈 Average Reward Progression (All Levels)",
    xaxis_title="Episode",
    yaxis_title="Average Reward",
    template='plotly_dark',
    hovermode='x unified',
    height=400
)

# 📊 نمودار 2: مقایسه Best Reward
fig_best = go.Figure()
for level_name, data in levels_data.items():
    if data:
        best_rewards = data.get('best_rewards', [])
        episodes = list(range(1, len(best_rewards) + 1))
        fig_best.add_trace(go.Scatter(
            x=episodes,
            y=best_rewards,
            mode='lines',
            name=level_name,
            line=dict(width=2, color=colors[level_name])
        ))

fig_best.update_layout(
    title="🏆 Best Reward Evolution (All Levels)",
    xaxis_title="Episode",
    yaxis_title="Best Reward",
    template='plotly_dark',
    hovermode='x unified',
    height=400
)

# 📊 نمودار 3: Actor Loss Comparison
fig_actor = go.Figure()
for level_name, data in levels_data.items():
    if data:
        actor_losses = data.get('actor_losses', [])
        episodes = list(range(1, len(actor_losses) + 1))
        fig_actor.add_trace(go.Scatter(
            x=episodes,
            y=actor_losses,
            mode='lines',
            name=level_name,
            line=dict(width=2, color=colors[level_name])
        ))

fig_actor.update_layout(
    title="🎭 Actor Loss Comparison",
    xaxis_title="Episode",
    yaxis_title="Actor Loss",
    template='plotly_dark',
    hovermode='x unified',
    height=400
)

# 📊 نمودار 4: Critic Loss Comparison  
fig_critic = go.Figure()
for level_name, data in levels_data.items():
    if data:
        critic_losses = data.get('critic_losses', [])
        episodes = list(range(1, len(critic_losses) + 1))
        fig_critic.add_trace(go.Scatter(
            x=episodes,
            y=critic_losses,
            mode='lines',
            name=level_name,
            line=dict(width=2, color=colors[level_name])
        ))

fig_critic.update_layout(
    title="🎯 Critic Loss Comparison",
    xaxis_title="Episode",
    yaxis_title="Critic Loss",
    template='plotly_dark',
    hovermode='x unified',
    height=400
)

# 📊 کارت‌های خلاصه
summary_cards = []
for level_name, data in levels_data.items():
    if data:
        rewards = data.get('average_rewards', [])
        best_rewards = data.get('best_rewards', [])
        
        final_avg = rewards[-1] if rewards else 0
        final_best = best_rewards[-1] if best_rewards else 0
        episodes = len(rewards)
        
        card = dbc.Card([
            dbc.CardHeader(html.H4(level_name, className="text-white")),
            dbc.CardBody([
                html.H5(f"Episodes: {episodes}", className="text-info"),
                html.H5(f"Final Avg Reward: {final_avg:.2f}", className="text-success"),
                html.H5(f"Best Reward: {final_best:.2f}", className="text-warning"),
            ])
        ], color=colors[level_name].replace('#', ''), outline=True, className="mb-3")
        summary_cards.append(card)

# 🎨 Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚁 MADDPG Training Results Dashboard", className="text-center text-info mb-4"),
            html.H4("Sequential Multi-Level Training Analysis", className="text-center text-muted mb-5")
        ])
    ]),
    
    # کارت‌های خلاصه
    dbc.Row([
        dbc.Col(card, width=4) for card in summary_cards
    ]),
    
    html.Hr(className="my-4"),
    
    # نمودارها
    dbc.Row([
        dbc.Col([dcc.Graph(figure=fig_reward)], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([dcc.Graph(figure=fig_best)], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([dcc.Graph(figure=fig_actor)], width=6),
        dbc.Col([dcc.Graph(figure=fig_critic)], width=6)
    ]),
    
    html.Hr(className="my-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.P("🎓 SkyMind Dashboard - MADDPG Sequential Training", 
                   className="text-center text-muted")
        ])
    ])
    
], fluid=True, style={'backgroundColor': '#0a0a0a', 'minHeight': '100vh', 'padding': '20px'})

if __name__ == '__main__':
    print("\n" + "="*70)
    print("✅ Dashboard is ready!")
    print("🌐 Open: http://127.0.0.1:8052")
    print("="*70 + "\n")
    app.run(debug=True, port=8052)
