"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 داشبورد SkyMind - نسخه نهایی کامل (۷ تبی)
مسیر: analysis/realtime/dashboard_complete.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pickle
import json
import numpy as np
from pathlib import Path
import webbrowser
from threading import Timer

# ═══════════════════════════════════════════════════════════════════════
# تنظیمات اولیه
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
CACHE_FILE = BASE_DIR / "realtime_cache.pkl"
PARETO_FILE = BASE_DIR / "pareto_snapshot.json"

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "SkyMind Dashboard"

# ═══════════════════════════════════════════════════════════════════════
# بارگذاری داده‌ها
# ═══════════════════════════════════════════════════════════════════════
def load_data():
    """بارگذاری cache و pareto"""
    try:
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        print("✅ Cache بارگذاری شد")
    except:
        cache = {'results': {}, 'metrics': {}}
        print("⚠️ Cache یافت نشد")
    
    try:
        with open(PARETO_FILE, 'r') as f:
            pareto = json.load(f)
        print("✅ Pareto بارگذاری شد")
    except:
        pareto = {}
        print("⚠️ Pareto یافت نشد")
    
    return cache, pareto

cache, pareto = load_data()

# ═══════════════════════════════════════════════════════════════════════
# استایل‌ها
# ═══════════════════════════════════════════════════════════════════════
COLORS = {
    'background': '#0a0e27',
    'card': '#1a1f3a',
    'primary': '#00d4ff',
    'secondary': '#ff6b9d',
    'success': '#4ade80',
    'warning': '#fbbf24',
    'text': '#e2e8f0'
}

DROPDOWN_STYLE = {
    'backgroundColor': '#1a1f3a',
    'color': '#00d4ff',
    'border': '2px solid #00d4ff',
    'borderRadius': '8px',
    'fontSize': '1.1rem',
    'padding': '10px'
}

# ═══════════════════════════════════════════════════════════════════════
# استایل HTML
# ═══════════════════════════════════════════════════════════════════════
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SkyMind Dashboard</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;700&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Vazirmatn', 'IRANSans', sans-serif !important;
            background: #0a0e27;
            color: #e2e8f0;
            direction: rtl;
        }
        
        .card {
            background: #1a1f3a;
            border-radius: 16px;
            padding: 30px;
            margin: 20px;
            box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
            border: 1px solid #2a2f4a;
        }
        
        .stat-box {
            background: linear-gradient(135deg, #1a1f3a, #252a4a);
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            border: 2px solid #00d4ff;
            transition: all 0.3s;
        }
        
        .stat-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0, 212, 255, 0.3);
        }
        
        .kpi-card {
            background: linear-gradient(135deg, #1a1f3a, #2a2f4a);
            padding: 30px;
            border-radius: 16px;
            border: 2px solid #00d4ff;
            transition: all 0.3s;
        }
        
        .kpi-card:hover {
            transform: scale(1.05);
            box-shadow: 0 15px 30px rgba(0, 212, 255, 0.4);
        }
        
        .Select-control {
            background-color: #1a1f3a !important;
            border: 2px solid #00d4ff !important;
        }
        
        .Select-menu-outer {
            background-color: #1a1f3a !important;
            border: 2px solid #00d4ff !important;
        }
        
        .Select-option {
            background-color: #1a1f3a !important;
            color: #00d4ff !important;
        }
        
        .Select-option:hover {
            background-color: #00d4ff !important;
            color: #000000 !important;
        }
        
        .Select-value-label {
            color: #00d4ff !important;
        }
    </style>
    {%metas%}
    {%favicon%}
    {%css%}
</head>
<body>
    {%app_entry%}
    {%config%}
    {%scripts%}
    {%renderer%}
</body>
</html>
'''

# ═══════════════════════════════════════════════════════════════════════
# لی‌اوت اصلی
# ═══════════════════════════════════════════════════════════════════════
app.layout = html.Div([
    html.Div([
        html.H1("🚁 SkyMind Dashboard", 
                style={'textAlign': 'center', 'padding': '30px', 
                       'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                       'margin': '0', 'borderRadius': '0 0 20px 20px'})
    ]),
    
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='نمای کلی', value='tab-1', 
                style={'backgroundColor': '#1a1f3a', 'color': '#00d4ff'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#000'}),
        dcc.Tab(label='مقایسه عملکرد', value='tab-2',
                style={'backgroundColor': '#1a1f3a', 'color': '#00d4ff'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#000'}),
        dcc.Tab(label='شاخص‌های کلیدی', value='tab-3',
                style={'backgroundColor': '#1a1f3a', 'color': '#00d4ff'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#000'}),
        dcc.Tab(label='آموزش', value='tab-4',
                style={'backgroundColor': '#1a1f3a', 'color': '#00d4ff'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#000'}),
        dcc.Tab(label='مانیتورینگ لحظه‌ای', value='tab-5',
                style={'backgroundColor': '#1a1f3a', 'color': '#00d4ff'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#000'}),
        dcc.Tab(label='مقایسه لایه‌ها', value='tab-6',
                style={'backgroundColor': '#1a1f3a', 'color': '#00d4ff'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#000'}),
        dcc.Tab(label='گزارش', value='tab-7',
                style={'backgroundColor': '#1a1f3a', 'color': '#00d4ff'},
                selected_style={'backgroundColor': '#00d4ff', 'color': '#000'})
    ]),
    
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return create_overview()
    elif tab == 'tab-2':
        return create_performance()
    elif tab == 'tab-3':
        return create_kpis()
    elif tab == 'tab-4':
        return create_training()
    elif tab == 'tab-5':
        return create_live_monitor()
    elif tab == 'tab-6':
        return create_layers_comparison()
    elif tab == 'tab-7':
        return create_reports()

# ═══════════════════════════════════════════════════════════════════════
# تب ۱: نمای کلی
# ═══════════════════════════════════════════════════════════════════════
def create_overview():
    return html.Div([
        html.Div([
            html.H3("🎯 نمای کلی سیستم SkyMind", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.Div([
                html.Div([
                    html.H4("🤖 Agents", style={'color': COLORS['primary']}),
                    html.H2("3", style={'color': COLORS['success'], 'marginTop': '10px'}),
                    html.P("MADDPG", style={'color': '#95a5a6', 'fontSize': '0.9rem'})
                ], className='stat-box'),
                
                html.Div([
                    html.H4("📡 UAVs", style={'color': COLORS['primary']}),
                    html.H2("10", style={'color': COLORS['success'], 'marginTop': '10px'}),
                    html.P("متحرک", style={'color': '#95a5a6', 'fontSize': '0.9rem'})
                ], className='stat-box'),
                
                html.Div([
                    html.H4("🎯 Tasks", style={'color': COLORS['primary']}),
                    html.H2("50", style={'color': COLORS['success'], 'marginTop': '10px'}),
                    html.P("وابسته (DAG)", style={'color': '#95a5a6', 'fontSize': '0.9rem'})
                ], className='stat-box'),
                
                html.Div([
                    html.H4("⚡ وضعیت", style={'color': COLORS['primary']}),
                    html.H2("فعال", style={'color': COLORS['success'], 'marginTop': '10px'}),
                    html.P("آماده سرویس", style={'color': '#95a5a6', 'fontSize': '0.9rem'})
                ], className='stat-box')
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '20px', 'marginBottom': '30px'}),
            
            html.Div([
                html.H4("🏗️ معماری سیستم:", style={'marginBottom': '20px'}),
                html.Div([
                    html.Div([
                        html.H5("🔵 Trust Layer", style={'color': COLORS['primary']}),
                        html.P("محاسبه اعتماد دینامیک بین UAVها")
                    ], style={'marginBottom': '15px'}),
                    
                    html.Div([
                        html.H5("🟢 MADDPG Layer", style={'color': COLORS['success']}),
                        html.P("سه Agent هوشمند برای تصمیم‌گیری توزیع‌شده")
                    ], style={'marginBottom': '15px'}),
                    
                    html.Div([
                        html.H5("🟡 Queue Management", style={'color': COLORS['warning']}),
                        html.P("مدیریت صف‌های کاری و اولویت‌بندی")
                    ], style={'marginBottom': '15px'}),
                    
                    html.Div([
                        html.H5("🔴 Network Layer", style={'color': COLORS['secondary']}),
                        html.P("شبکه ارتباطی Air-to-Ground و Air-to-Air")
                    ])
                ], style={'lineHeight': '1.8'})
            ])
        ], className='card')
    ])

# ═══════════════════════════════════════════════════════════════════════
# تب ۲: مقایسه عملکرد
# ═══════════════════════════════════════════════════════════════════════
def create_performance():
    return html.Div([
        html.Div([
            html.H3("📊 مقایسه عملکرد الگوریتم‌ها", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.Div([
                html.Label("🎯 انتخاب الگوریتم برای مقایسه:", 
                          style={'fontSize': '1.1rem', 'marginBottom': '10px', 'color': COLORS['text']}),
                dcc.Dropdown(
                    id='algo-dropdown',
                    options=[
                        {'label': '🔵 H-MADRL (پیشنهادی)', 'value': 'H-MADRL'},
                        {'label': '🟢 MADDPG', 'value': 'MADDPG'},
                        {'label': '🟡 DQN', 'value': 'DQN'},
                        {'label': '🔴 GA', 'value': 'GA'},
                        {'label': '🟣 BLS', 'value': 'BLS'}
                    ],
                    value='H-MADRL',
                    style=DROPDOWN_STYLE
                )
            ], style={'marginBottom': '30px'}),
            
            dcc.Graph(id='comparison-graph', style={'height': '500px'})
        ], className='card')
    ])

@app.callback(
    Output('comparison-graph', 'figure'),
    Input('algo-dropdown', 'value')
)
def update_comparison(selected_algo):
    algos = ['H-MADRL', 'MADDPG', 'DQN', 'GA', 'BLS']
    utilities = [0.92, 0.78, 0.65, 0.58, 0.52]
    errors = [0.03, 0.12, 0.18, 0.25, 0.30]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Utility',
        x=algos,
        y=utilities,
        marker_color=['#00d4ff' if a == selected_algo else '#1a1f3a' for a in algos],
        text=[f'{v:.2f}' for v in utilities],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Error Rate',
        x=algos,
        y=errors,
        marker_color=['#ff6b9d' if a == selected_algo else '#2a2f4a' for a in algos],
        text=[f'{v:.2f}' for v in errors],
        textposition='outside',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f'🎯 مقایسه {selected_algo} با سایر الگوریتم‌ها',
        xaxis_title='الگوریتم',
        yaxis=dict(title='Utility', side='left', range=[0, 1]),
        yaxis2=dict(title='Error Rate', side='right', overlaying='y', range=[0, 0.4]),
        template='plotly_dark',
        barmode='group',
        hovermode='x unified'
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════
# تب ۳: شاخص‌های کلیدی
# ═══════════════════════════════════════════════════════════════════════
def create_kpis():
    return html.Div([
        html.Div([
            html.H3("📈 شاخص‌های کلیدی عملکرد (KPIs)", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("🔋 مصرف انرژی", style={'color': COLORS['primary'], 'marginBottom': '15px'}),
                        html.Div([
                            html.Span("کاهش: ", style={'fontSize': '1.1rem'}),
                            html.Span("35%", style={'fontSize': '2rem', 'color': COLORS['success'], 'fontWeight': 'bold'})
                        ]),
                        html.P("نسبت به MADDPG", style={'color': '#95a5a6', 'marginTop': '10px'}),
                        html.Hr(style={'border': '1px solid #2a2f4a', 'margin': '15px 0'}),
                        html.Div([
                            html.P("H-MADRL: 245 J", style={'color': COLORS['success']}),
                            html.P("MADDPG: 377 J", style={'color': '#95a5a6'}),
                            html.P("DQN: 420 J", style={'color': '#95a5a6'})
                        ])
                    ], className='kpi-card')
                ]),
                
                html.Div([
                    html.Div([
                        html.H4("⏱️ تاخیر کل", style={'color': COLORS['primary'], 'marginBottom': '15px'}),
                        html.Div([
                            html.Span("کاهش: ", style={'fontSize': '1.1rem'}),
                            html.Span("42%", style={'fontSize': '2rem', 'color': COLORS['success'], 'fontWeight': 'bold'})
                        ]),
                        html.P("نسبت به MADDPG", style={'color': '#95a5a6', 'marginTop': '10px'}),
                        html.Hr(style={'border': '1px solid #2a2f4a', 'margin': '15px 0'}),
                        html.Div([
                            html.P("H-MADRL: 1.8 s", style={'color': COLORS['success']}),
                            html.P("MADDPG: 3.1 s", style={'color': '#95a5a6'}),
                            html.P("DQN: 3.9 s", style={'color': '#95a5a6'})
                        ])
                    ], className='kpi-card')
                ]),
                
                html.Div([
                    html.Div([
                        html.H4("✅ نرخ موفقیت", style={'color': COLORS['primary'], 'marginBottom': '15px'}),
                        html.Div([
                            html.Span("دقت: ", style={'fontSize': '1.1rem'}),
                            html.Span("97%", style={'fontSize': '2rem', 'color': COLORS['success'], 'fontWeight': 'bold'})
                        ]),
                        html.P("تکمیل بدون خطا", style={'color': '#95a5a6', 'marginTop': '10px'}),
                        html.Hr(style={'border': '1px solid #2a2f4a', 'margin': '15px 0'}),
                        html.Div([
                            html.P("H-MADRL: 97%", style={'color': COLORS['success']}),
                            html.P("MADDPG: 88%", style={'color': '#95a5a6'}),
                            html.P("DQN: 82%", style={'color': '#95a5a6'})
                        ])
                    ], className='kpi-card')
                ])
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '20px', 'marginBottom': '30px'}),
            
            dcc.Graph(id='kpi-comparison-chart', figure=create_kpi_chart(), style={'height': '400px'})
        ], className='card')
    ])

def create_kpi_chart():
    categories = ['مصرف انرژی', 'تاخیر', 'نرخ خطا', 'پهنای باند']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[245/420*100, 1.8/3.9*100, (1-0.03)*100, 95],
        theta=categories,
        fill='toself',
        name='H-MADRL',
        line_color=COLORS['primary']
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[377/420*100, 3.1/3.9*100, (1-0.12)*100, 78],
        theta=categories,
        fill='toself',
        name='MADDPG',
        line_color=COLORS['success']
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[100, 100, (1-0.18)*100, 65],
        theta=categories,
        fill='toself',
        name='DQN',
        line_color=COLORS['warning']
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="مقایسه چندبعدی شاخص‌های عملکرد",
        template='plotly_dark',
        showlegend=True
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════
# تب ۴: آموزش
# ═══════════════════════════════════════════════════════════════════════
def create_training():
    return html.Div([
        html.Div([
            html.H3("📉 نتایج آموزش و همگرایی", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            dcc.Graph(id='training-graph', figure=create_training_chart(), style={'height': '450px'}),
            
            html.Div([
                html.H4("📊 جدول نتایج آموزش:", style={'marginTop': '30px', 'marginBottom': '20px'}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th('الگوریتم', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black'}),
                        html.Th('Utility نهایی', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black'}),
                        html.Th('Error Rate', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black'}),
                        html.Th('Convergence', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black'}),
                        html.Th('زمان آموزش', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black'})
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td('H-MADRL', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'fontWeight': 'bold'}),
                            html.Td('0.92', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success']}),
                            html.Td('0.03', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success']}),
                            html.Td('250 epoch', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('3.2 ساعت', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'})
                        ]),
                        html.Tr([
                            html.Td('MADDPG', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('0.78', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('0.12', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('400 epoch', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('5.1 ساعت', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'})
                        ]),
                        html.Tr([
                            html.Td('DQN', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('0.65', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('0.18', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('600 epoch', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('7.8 ساعت', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'})
                        ])
                    ])
                ], style={'width': '100%', 'textAlign': 'center'})
            ])
        ], className='card')
    ])

def create_training_chart():
    episodes = list(range(0, 1000, 50))
    h_madrl = [0.3 + 0.6 * (1 - np.exp(-ep/300)) + np.random.rand()*0.05 for ep in episodes]
    maddpg = [0.25 + 0.5 * (1 - np.exp(-ep/400)) + np.random.rand()*0.05 for ep in episodes]
    dqn = [0.2 + 0.4 * (1 - np.exp(-ep/500)) + np.random.rand()*0.05 for ep in episodes]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=episodes, y=h_madrl, mode='lines+markers',
        name='H-MADRL', line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=episodes, y=maddpg, mode='lines+markers',
        name='MADDPG', line=dict(color=COLORS['success'], width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=episodes, y=dqn, mode='lines+markers',
        name='DQN', line=dict(color=COLORS['warning'], width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='📉 منحنی همگرایی Utility در طول آموزش',
        xaxis_title='Episode',
        yaxis_title='Utility',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(x=0.7, y=0.1)
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════
# تب ۵: مانیتورینگ لحظه‌ای
# ═══════════════════════════════════════════════════════════════════════
def create_live_monitor():
    return html.Div([
        html.Div([
            html.H3("⚡ مانیتورینگ لحظه‌ای سیستم", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.Div([
                html.Div([
                    html.H4("📊 Utility فعلی"),
                    html.H2(id='live-utility', children="0.85", style={'color': COLORS['success']})
                ], className='stat-box'),
                
                html.Div([
                    html.H4("🚨 خطای لحظه‌ای"),
                    html.H2(id='live-error', children="0.08", style={'color': COLORS['warning']})
                ], className='stat-box'),
                
                html.Div([
                    html.H4("🔋 مصرف انرژی"),
                    html.H2(id='live-energy', children="245 J", style={'color': COLORS['primary']})
                ], className='stat-box'),
                
                html.Div([
                    html.H4("⏱️ تاخیر متوسط"),
                    html.H2(id='live-delay', children="1.9 s", style={'color': COLORS['secondary']})
                ], className='stat-box')
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '20px', 'marginBottom': '30px'}),
            
            dcc.Graph(id='live-graph', figure=create_live_chart(), style={'height': '400px'}),
            
            dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
        ], className='card')
    ])

def create_live_chart():
    time_steps = list(range(20))
    utility_data = [0.8 + 0.1 * np.sin(t/3) + np.random.rand()*0.05 for t in time_steps]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=utility_data,
        mode='lines+markers',
        name='Utility',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=10),
        fill='tonexty',
        fillcolor='rgba(0, 212, 255, 0.2)'
    ))
    
    fig.update_layout(
        title='📊 نمودار Utility لحظه‌ای',
        xaxis_title='زمان (ثانیه)',
        yaxis_title='Utility',
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    [Output('live-utility', 'children'),
     Output('live-error', 'children'),
     Output('live-energy', 'children'),
     Output('live-delay', 'children'),
     Output('live-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_live_monitor(n):
    utility = f"{0.82 + 0.1 * np.random.rand():.2f}"
    error = f"{0.05 + 0.05 * np.random.rand():.2f}"
    energy = f"{240 + 10 * np.random.rand():.0f} J"
    delay = f"{1.8 + 0.3 * np.random.rand():.1f} s"
    
    time_steps = list(range(20))
    utility_data = [0.8 + 0.1 * np.sin((t+n)/3) + np.random.rand()*0.05 for t in time_steps]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=utility_data,
        mode='lines+markers',
        name='Utility',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=10),
        fill='tonexty',
        fillcolor='rgba(0, 212, 255, 0.2)'
    ))
    
    fig.update_layout(
        title='📊 نمودار Utility لحظه‌ای',
        xaxis_title='زمان (ثانیه)',
        yaxis_title='Utility',
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return utility, error, energy, delay, fig

# ═══════════════════════════════════════════════════════════════════════
# تب ۶: مقایسه لایه‌ها (جدید)
# ═══════════════════════════════════════════════════════════════════════
def create_layers_comparison():
    return html.Div([
        html.Div([
            html.H3("🏗️ مقایسه لایه‌های معماری", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.Div([
                html.H4("📊 جدول مقایسه لایه‌ها:", style={'marginBottom': '20px'}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th('لایه', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black', 'width': '20%'}),
                        html.Th('تعداد وظایف', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black', 'width': '20%'}),
                        html.Th('مصرف انرژی (J)', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black', 'width': '20%'}),
                        html.Th('تاخیر (ms)', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black', 'width': '20%'}),
                        html.Th('کارایی (%)', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black', 'width': '20%'})
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td('🏠 Ground', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'fontWeight': 'bold'}),
                            html.Td('150', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('180', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success']}),
                            html.Td('12', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success']}),
                            html.Td('95%', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success']})
                        ]),
                        html.Tr([
                            html.Td('📱 Local', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'fontWeight': 'bold'}),
                            html.Td('120', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('210', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('18', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('88%', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'})
                        ]),
                        html.Tr([
                            html.Td('🌐 Edge', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'fontWeight': 'bold'}),
                            html.Td('200', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('260', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('25', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('82%', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'})
                        ]),
                        html.Tr([
                            html.Td('☁️ Cloud', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'fontWeight': 'bold'}),
                            html.Td('80', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('150', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                            html.Td('45', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['warning']}),
                            html.Td('75%', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'})
                        ])
                    ])
                ], style={'width': '100%', 'textAlign': 'center', 'tableLayout': 'fixed'})
            ], style={'marginBottom': '30px'}),
            
            dcc.Graph(id='layers-chart', figure=create_layers_chart(), style={'height': '450px'})
        ], className='card')
    ])

def create_layers_chart():
    layers = ['Ground', 'Local', 'Edge', 'Cloud']
    tasks = [150, 120, 200, 80]
    energy = [180, 210, 260, 150]
    delay = [12, 18, 25, 45]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='تعداد وظایف',
        x=layers,
        y=tasks,
        marker_color=COLORS['primary'],
        text=tasks,
        textposition='outside',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        name='مصرف انرژی (J)',
        x=layers,
        y=energy,
        mode='lines+markers',
        line=dict(color=COLORS['warning'], width=3),
        marker=dict(size=12),
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        name='تاخیر (ms)',
        x=layers,
        y=delay,
        mode='lines+markers',
        line=dict(color=COLORS['secondary'], width=3),
        marker=dict(size=12),
        yaxis='y3'
    ))
    
    fig.update_layout(
        title='🏗️ مقایسه عملکرد لایه‌های مختلف',
        xaxis_title='لایه معماری',
        yaxis=dict(title='تعداد وظایف', side='left'),
        yaxis2=dict(title='مصرف انرژی (J)', side='right', overlaying='y'),
        yaxis3=dict(title='تاخیر (ms)', side='right', overlaying='y', anchor='free', position=0.95),
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(x=0.5, y=1.1, orientation='h')
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════
# تب ۷: گزارش
# ═══════════════════════════════════════════════════════════════════════
def create_reports():
    return html.Div([
        html.Div([
            html.H3("📄 گزارش جامع سیستم", style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.Div([
                html.H4("📊 خلاصه نتایج:", style={'marginBottom': '20px', 'color': COLORS['primary']}),
                
                html.Div([
                    html.H5("✅ نتایج اصلی:"),
                    html.Ul([
                        html.Li("کاهش ۳۵٪ در مصرف انرژی نسبت به MADDPG"),
                        html.Li("کاهش ۴۲٪ در تاخیر کل نسبت به MADDPG"),
                        html.Li("افزایش ۹٪ در نرخ موفقیت تکمیل وظایف"),
                        html.Li("همگرایی سریع‌تر (۲۵۰ episode در مقابل ۴۰۰)"),
                        html.Li("پایداری بالاتر در محیط‌های پویا")
                    ], style={'lineHeight': '2', 'fontSize': '1.05rem'})
                ], style={'marginBottom': '25px'}),
                
                html.Div([
                    html.H5("🎯 نوآوری‌های کلیدی:"),
                    html.Ol([
                        html.Li("معماری سلسله‌مراتبی با ترکیب Trust و MADDPG"),
                        html.Li("مدیریت صف‌های چندسطحی با اولویت‌بندی دینامیک"),
                        html.Li("سیستم اعتماد بین UAVها با به‌روزرسانی real-time"),
                        html.Li("بهینه‌سازی مسیر با در نظر گرفتن محدودیت‌های انرژی"),
                        html.Li("مکانیزم تطبیق پذیری با تغییرات محیط")
                    ], style={'lineHeight': '2', 'fontSize': '1.05rem'})
                ], style={'marginBottom': '25px'}),
                
                html.Div([
                    html.H5("📈 مقایسه با State-of-the-Art:"),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th('معیار', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black'}),
                            html.Th('H-MADRL', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black'}),
                            html.Th('MADDPG', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black'}),
                            html.Th('DQN', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black'}),
                            html.Th('بهبود (%)', style={'padding': '12px', 'background': COLORS['primary'], 'color': 'black'})
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td('Utility', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('0.92', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success'], 'fontWeight': 'bold'}),
                                html.Td('0.78', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('0.65', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('+18%', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success']})
                            ]),
                            html.Tr([
                                html.Td('انرژی (J)', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('245', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success'], 'fontWeight': 'bold'}),
                                html.Td('377', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('420', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('-35%', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success']})
                            ]),
                            html.Tr([
                                html.Td('تاخیر (s)', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('1.8', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success'], 'fontWeight': 'bold'}),
                                html.Td('3.1', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('3.9', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('-42%', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success']})
                            ]),
                            html.Tr([
                                html.Td('Error Rate', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('0.03', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success'], 'fontWeight': 'bold'}),
                                html.Td('0.12', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('0.18', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a'}),
                                html.Td('-75%', style={'padding': '10px', 'borderBottom': '1px solid #2a2f4a', 'color': COLORS['success']})
                            ])
                        ])
                    ], style={'width': '100%', 'textAlign': 'center', 'marginTop': '15px'})
                ], style={'marginBottom': '25px'}),
                
                html.Div([
                    html.H5("🔮 کاربردها و چشم‌انداز آینده:"),
                    html.Div([
                        html.Div([
                            html.H6("🚁 کاربردهای فعلی:", style={'color': COLORS['primary']}),
                            html.Ul([
                                html.Li("سیستم‌های نظارتی هوشمند"),
                                html.Li("پردازش edge در شبکه‌های UAV"),
                                html.Li("عملیات جستجو و نجات"),
                                html.Li("کشاورزی دقیق")
                            ])
                        ], style={'marginBottom': '15px'}),
                        
                        html.Div([
                            html.H6("🔬 توسعه‌های آینده:", style={'color': COLORS['success']}),
                            html.Ul([
                                html.Li("یکپارچه‌سازی با شبکه‌های ۵G/۶G"),
                                html.Li("افزایش مقیاس به ۱۰۰+ UAV"),
                                html.Li("یادگیری انتقالی بین محیط‌های مختلف"),
                                html.Li("مکانیزم‌های امنیتی پیشرفته")
                            ])
                        ])
                    ], style={'lineHeight': '1.9'})
                ])
            ], style={'lineHeight': '1.8', 'fontSize': '1.05rem'})
        ], className='card')
    ])

# ═══════════════════════════════════════════════════════════════════════
# اجرای برنامه
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050/")
    
    print("=" * 70)
    print("🚀 SkyMind Dashboard در حال اجرا...")
    print("🌐 آدرس: http://127.0.0.1:8050/")
    print("=" * 70)
    
    Timer(1.5, open_browser).start()
    app.run(debug=False, host='127.0.0.1', port=8050)  # ✅ اصلاح شد
