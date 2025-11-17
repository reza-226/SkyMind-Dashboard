# dashboard/layouts/multi_env_tab.py

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc
import json
from pathlib import Path

def load_all_results():
    """بارگذاری نتایج هر سه سطح"""
    levels = {
        'level1_simple': {'name': 'ساده (بدون مانع)', 'color': '#00D9FF'},
        'level2_medium': {'name': 'متوسط (2 مانع)', 'color': '#FFA500'},
        'level3_complex': {'name': 'پیچیده (4 مانع)', 'color': '#FF4444'}
    }
    
    results = {}
    for level_id, meta in levels.items():
        result_file = Path('models') / level_id / 'training_results.json'
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results[level_id] = {
                    **data,
                    'display_name': meta['name'],
                    'color': meta['color']
                }
    
    return results

def create_reward_comparison_chart(results):
    """نمودار مقایسه‌ای Reward"""
    fig = go.Figure()
    
    for level_id, data in results.items():
        if 'episode_rewards' in data['results']:
            rewards = data['results']['episode_rewards']
            episodes = list(range(1, len(rewards) + 1))
            
            # هموارسازی با Moving Average
            window = 20
            smoothed = []
            for i in range(len(rewards)):
                start = max(0, i - window)
                smoothed.append(sum(rewards[start:i+1]) / (i - start + 1))
            
            fig.add_trace(go.Scatter(
                x=episodes,
                y=smoothed,
                mode='lines',
                name=data['display_name'],
                line=dict(color=data['color'], width=3),
                hovertemplate='<b>Episode:</b> %{x}<br><b>Reward:</b> %{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title={
            'text': '📈 مقایسه پیشرفت یادگیری در سطوح مختلف',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Episode",
        yaxis_title="Average Reward (Smoothed)",
        height=500,
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        ),
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white')
    )
    
    return fig

def create_best_reward_bar_chart(results):
    """نمودار میله‌ای بهترین Reward"""
    labels = []
    best_rewards = []
    colors = []
    
    for level_id, data in results.items():
        labels.append(data['display_name'])
        best_rewards.append(data['results']['best_reward'])
        colors.append(data['color'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=best_rewards,
            marker_color=colors,
            text=[f"{r:.2f}" for r in best_rewards],
            textposition='outside',
            textfont=dict(size=16, color='white')
        )
    ])
    
    fig.update_layout(
        title={
            'text': '🏆 مقایسه بهترین Reward',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        yaxis_title="Best Reward",
        height=400,
        template='plotly_dark',
        showlegend=False,
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white')
    )
    
    return fig

def create_convergence_chart(results):
    """نمودار سرعت همگرایی"""
    convergence_data = []
    
    for level_id, data in results.items():
        rewards = data['results'].get('episode_rewards', [])
        threshold = -50  # حد آستانه همگرایی
        
        converge_ep = next((i+1 for i, r in enumerate(rewards) if r > threshold), len(rewards))
        
        convergence_data.append({
            'Level': data['display_name'],
            'Episodes': converge_ep,
            'Color': data['color']
        })
    
    df = pd.DataFrame(convergence_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Level'],
            y=df['Episodes'],
            marker_color=df['Color'],
            text=df['Episodes'],
            textposition='outside',
            textfont=dict(size=16, color='white')
        )
    ])
    
    fig.update_layout(
        title={
            'text': f'⏱️ سرعت همگرایی (Threshold: {threshold})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        yaxis_title="Episodes to Converge",
        height=350,
        template='plotly_dark',
        showlegend=False,
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white')
    )
    
    return fig

def create_variance_chart(results):
    """نمودار واریانس (پایداری)"""
    import numpy as np
    
    variance_data = []
    
    for level_id, data in results.items():
        rewards = data['results'].get('episode_rewards', [])
        if len(rewards) >= 100:
            variance = np.var(rewards[-100:])  # واریانس 100 اپیزود آخر
            variance_data.append({
                'Level': data['display_name'],
                'Variance': variance,
                'Color': data['color']
            })
    
    df = pd.DataFrame(variance_data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Level'],
            y=df['Variance'],
            marker_color=df['Color'],
            text=[f"{v:.2f}" for v in df['Variance']],
            textposition='outside',
            textfont=dict(size=16, color='white')
        )
    ])
    
    fig.update_layout(
        title={
            'text': '📊 واریانس Reward (پایداری یادگیری)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'white'}
        },
        yaxis_title="Variance (Last 100 Episodes)",
        height=350,
        template='plotly_dark',
        showlegend=False,
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2d2d2d',
        font=dict(color='white')
    )
    
    return fig

def layout(df=None):
    """Layout اصلی تب Multi-Environment"""
    
    # بارگذاری نتایج
    results = load_all_results()
    
    if not results:
        return html.Div([
            html.H2("⚠️ هیچ نتیجه‌ای یافت نشد", 
                    style={'textAlign': 'center', 'color': '#FFA500', 'marginTop': '50px'}),
            html.P("ابتدا Training را با دستور زیر انجام دهید:",
                   style={'textAlign': 'center', 'color': 'white', 'fontSize': '18px'}),
            html.Code("python train_sequential_levels.py",
                     style={
                         'display': 'block',
                         'textAlign': 'center',
                         'backgroundColor': '#333',
                         'padding': '15px',
                         'borderRadius': '8px',
                         'color': '#00D9FF',
                         'fontSize': '16px',
                         'marginTop': '20px'
                     })
        ], style={'backgroundColor': '#1e1e1e', 'padding': '40px', 'minHeight': '80vh'})
    
    # ساخت Layout با نمودارها
    return html.Div([
        
        # Header
        html.Div([
            html.H2("🌐 مقایسه نتایج چند محیط", 
                    style={'color': 'white', 'textAlign': 'center'}),
            html.P("تحلیل Transfer Learning از محیط ساده به پیچیده",
                   style={'color': '#aaa', 'textAlign': 'center', 'fontSize': '16px'})
        ], style={'marginBottom': '30px'}),
        
        # کارت‌های خلاصه
        html.Div([
            html.Div([
                html.Div([
                    html.H4(data['display_name'], 
                           style={'color': data['color'], 'textAlign': 'center'}),
                    html.H2(f"{data['results']['best_reward']:.2f}",
                           style={'color': 'white', 'textAlign': 'center', 'margin': '10px 0'}),
                    html.P(f"🎯 {data['config']['training']['max_episodes']} Episodes",
                          style={'color': '#aaa', 'textAlign': 'center'})
                ], style={
                    'backgroundColor': '#2d2d2d',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'border': f'2px solid {data["color"]}'
                })
            ], style={'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'})
            for level_id, data in results.items()
        ], style={'marginBottom': '40px', 'textAlign': 'center'}),
        
        # نمودار اصلی مقایسه
        dcc.Graph(
            figure=create_reward_comparison_chart(results),
            config={'displayModeBar': False}
        ),
        
        html.Hr(style={'border': '1px solid #444', 'margin': '40px 0'}),
        
        # ردیف دوم: بهترین Reward + همگرایی
        html.Div([
            html.Div([
                dcc.Graph(
                    figure=create_best_reward_bar_chart(results),
                    config={'displayModeBar': False}
                )
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(
                    figure=create_convergence_chart(results),
                    config={'displayModeBar': False}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
        ]),
        
        html.Hr(style={'border': '1px solid #444', 'margin': '40px 0'}),
        
        # نمودار واریانس
        dcc.Graph(
            figure=create_variance_chart(results),
            config={'displayModeBar': False},
            style={'marginBottom': '40px'}
        ),
        
        # جدول مقایسه
        html.Div([
            html.H3("📊 جدول مقایسه تفصیلی", 
                   style={'color': 'white', 'marginBottom': '20px'}),
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th('سطح', style={'color': 'white', 'padding': '10px', 'borderBottom': '2px solid #444'}),
                        html.Th('Best Reward', style={'color': 'white', 'padding': '10px', 'borderBottom': '2px solid #444'}),
                        html.Th('Episodes', style={'color': 'white', 'padding': '10px', 'borderBottom': '2px solid #444'}),
                        html.Th('موانع', style={'color': 'white', 'padding': '10px', 'borderBottom': '2px solid #444'})
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(data['display_name'], style={'color': data['color'], 'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(f"{data['results']['best_reward']:.2f}", style={'color': 'white', 'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(data['config']['training']['max_episodes'], style={'color': 'white', 'padding': '10px', 'borderBottom': '1px solid #333'}),
                        html.Td(data['config']['params'].get('num_obstacles', 0), style={'color': 'white', 'padding': '10px', 'borderBottom': '1px solid #333'})
                    ])
                    for level_id, data in results.items()
                ])
            ], style={'width': '100%', 'backgroundColor': '#2d2d2d', 'borderRadius': '10px', 'overflow': 'hidden'})
        ])
        
    ], style={
        'backgroundColor': '#1e1e1e',
        'padding': '30px',
        'minHeight': '100vh'
    })
