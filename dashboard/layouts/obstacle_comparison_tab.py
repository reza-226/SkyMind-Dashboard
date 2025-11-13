"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎨 تب مقایسه موانع در داشبورد
مسیر: dashboard/layouts/obstacle_comparison_tab.py (NEW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from dash import dcc, html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def create_obstacle_comparison_layout():
    """
    ایجاد لایوت تب مقایسه موانع
    """
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 📊 داده‌های نمونه (در واقعیت از فایل می‌خوانیم)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # تولید داده نمونه
    np.random.seed(42)
    
    complexities = ['Simple', 'Medium', 'Complex']
    algorithms = ['MADDPG', 'DQN', 'BLS', 'GA', 'ECORI']
    layers = ['Ground', 'Local', 'Edge', 'Cloud']
    
    data = []
    for complexity in complexities:
        diff_factor = {'Simple': 1.0, 'Medium': 1.5, 'Complex': 2.2}[complexity]
        for algo in algorithms:
            algo_factor = {
                'MADDPG': 0.75, 'DQN': 1.0, 'BLS': 1.4, 
                'GA': 1.6, 'ECORI': 0.85
            }[algo]
            for layer in layers:
                layer_factor = {
                    'Ground': 1.3, 'Local': 1.15, 
                    'Edge': 0.85, 'Cloud': 1.0
                }[layer]
                
                data.append({
                    'Complexity': complexity,
                    'Algorithm': algo,
                    'Layer': layer,
                    'Delay': np.random.uniform(40, 120) * diff_factor * algo_factor * layer_factor,
                    'Energy': np.random.uniform(8, 40) * diff_factor * algo_factor,
                    'Success_Rate': max(65, 100 - np.random.uniform(3, 18) * diff_factor * algo_factor),
                    'Collision_Rate': min(25, np.random.uniform(0.5, 8) * diff_factor / algo_factor),
                    'Path_Length': np.random.uniform(180, 450) * diff_factor,
                    'Safety_Score': max(70, 100 - np.random.uniform(2, 15) * diff_factor / algo_factor)
                })
    
    df = pd.DataFrame(data)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🎨 ایجاد نمودارها
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # 1️⃣ نمودار Heatmap: الگوریتم × پیچیدگی
    heatmap_data = df.groupby(['Algorithm', 'Complexity'])['Delay'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Algorithm', columns='Complexity', values='Delay')
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='YlOrRd',
        text=np.round(heatmap_pivot.values, 2),
        texttemplate='%{text:.1f}',
        textfont={"size": 11},
        colorbar=dict(title="تأخیر (ms)")
    ))
    
    fig_heatmap.update_layout(
        title={
            'text': '🌡️ نقشه حرارتی: تأخیر میانگین بر حسب الگوریتم و پیچیدگی',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'family': 'Arial, sans-serif'}
        },
        xaxis_title='سطح پیچیدگی موانع',
        yaxis_title='الگوریتم',
        height=400,
        template='plotly_white'
    )
    
    # 2️⃣ نمودار میله‌ای گروهی: مقایسه سطوح پیچیدگی
    fig_bar_comparison = go.Figure()
    
    for complexity in complexities:
        df_comp = df[df['Complexity'] == complexity].groupby('Algorithm')['Delay'].mean()
        fig_bar_comparison.add_trace(go.Bar(
            name=complexity,
            x=df_comp.index,
            y=df_comp.values,
            text=np.round(df_comp.values, 1),
            textposition='outside'
        ))
    
    fig_bar_comparison.update_layout(
        title='📊 مقایسه تأخیر الگوریتم‌ها در سطوح مختلف پیچیدگی',
        xaxis_title='الگوریتم',
        yaxis_title='تأخیر میانگین (ms)',
        barmode='group',
        height=450,
        template='plotly_white',
        legend=dict(title='سطح پیچیدگی', orientation='h', y=1.1)
    )
    
    # 3️⃣ نمودار خطی: تأثیر افزایش پیچیدگی
    fig_line_trend = go.Figure()
    
    for algo in algorithms:
        df_algo = df[df['Algorithm'] == algo].groupby('Complexity').agg({
            'Delay': 'mean',
            'Collision_Rate': 'mean'
        }).reset_index()
        
        fig_line_trend.add_trace(go.Scatter(
            name=algo,
            x=df_algo['Complexity'],
            y=df_algo['Delay'],
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(width=2.5)
        ))
    
    fig_line_trend.update_layout(
        title='📈 روند تغییرات تأخیر با افزایش پیچیدگی موانع',
        xaxis_title='سطح پیچیدگی',
        yaxis_title='تأخیر میانگین (ms)',
        height=450,
        template='plotly_white',
        legend=dict(title='الگوریتم')
    )
    
    # 4️⃣ نمودار Box Plot: توزیع نرخ برخورد
    fig_box_collision = go.Figure()
    
    for complexity in complexities:
        df_comp = df[df['Complexity'] == complexity]
        fig_box_collision.add_trace(go.Box(
            name=complexity,
            y=df_comp['Collision_Rate'],
            boxmean='sd',
            marker_color=['green', 'orange', 'red'][complexities.index(complexity)]
        ))
    
    fig_box_collision.update_layout(
        title='🎯 توزیع نرخ برخورد در سطوح مختلف پیچیدگی',
        yaxis_title='نرخ برخورد (%)',
        height=400,
        template='plotly_white',
        showlegend=False
    )
    
    # 5️⃣ نمودار رادار: مقایسه چندبعدی لایه‌ها (Complex)
    df_complex = df[df['Complexity'] == 'Complex'].groupby('Layer').agg({
        'Delay': lambda x: 100 - (x.mean() / df['Delay'].max() * 100),
        'Energy': lambda x: 100 - (x.mean() / df['Energy'].max() * 100),
        'Success_Rate': 'mean',
        'Safety_Score': 'mean'
    }).reset_index()
    
    categories = ['تأخیر↓', 'انرژی↓', 'موفقیت↑', 'ایمنی↑']
    
    fig_radar_layers = go.Figure()
    
    for _, row in df_complex.iterrows():
        values = [
            row['Delay'],
            row['Energy'],
            row['Success_Rate'],
            row['Safety_Score']
        ]
        values += values[:1]  # بستن چندضلعی
        
        fig_radar_layers.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=row['Layer'],
            line=dict(width=2)
        ))
    
    fig_radar_layers.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title='🎯 مقایسه چندبعدی لایه‌ها (سناریوی پیچیده)',
        height=500,
        template='plotly_white'
    )
    
    # 6️⃣ نمودار Scatter: Success Rate vs Collision Rate
    fig_scatter = px.scatter(
        df,
        x='Collision_Rate',
        y='Success_Rate',
        color='Algorithm',
        size='Delay',
        facet_col='Complexity',
        hover_data=['Layer'],
        title='🔍 نرخ موفقیت در برابر نرخ برخورد',
        labels={
            'Collision_Rate': 'نرخ برخورد (%)',
            'Success_Rate': 'نرخ موفقیت (%)'
        },
        height=450,
        template='plotly_white'
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🏗️ ساخت لایوت
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    layout = html.Div([
        # Header
        html.Div([
            html.H2('🚧 مقایسه جامع عملکرد در سطوح مختلف موانع',
                   style={
                       'textAlign': 'center',
                       'color': '#2c3e50',
                       'marginBottom': '10px',
                       'fontFamily': 'Arial, sans-serif'
                   }),
            html.P('تحلیل تأثیر پیچیدگی موانع بر عملکرد الگوریتم‌ها و لایه‌های محاسباتی',
                  style={
                      'textAlign': 'center',
                      'color': '#7f8c8d',
                      'fontSize': '14px',
                      'marginBottom': '25px'
                  })
        ], className='header-section'),
        
        # کارت‌های آماری
        html.Div([
            html.Div([
                html.Div([
                    html.I(className='fas fa-layer-group', 
                          style={'fontSize': '28px', 'color': '#3498db'}),
                    html.H4('3', style={'margin': '10px 0 5px 0', 'fontSize': '32px'}),
                    html.P('سطح پیچیدگی', style={'margin': 0, 'color': '#7f8c8d', 'fontSize': '13px'})
                ], className='stat-card', style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'color': 'white'
                })
            ], className='col-md-3'),
            
            html.Div([
                html.Div([
                    html.I(className='fas fa-brain', 
                          style={'fontSize': '28px', 'color': '#e74c3c'}),
                    html.H4('5', style={'margin': '10px 0 5px 0', 'fontSize': '32px'}),
                    html.P('الگوریتم', style={'margin': 0, 'color': '#7f8c8d', 'fontSize': '13px'})
                ], className='stat-card', style={
                    'background': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                    'color': 'white'
                })
            ], className='col-md-3'),
            
            html.Div([
                html.Div([
                    html.I(className='fas fa-server', 
                          style={'fontSize': '28px', 'color': '#27ae60'}),
                    html.H4('4', style={'margin': '10px 0 5px 0', 'fontSize': '32px'}),
                    html.P('لایه محاسباتی', style={'margin': 0, 'color': '#7f8c8d', 'fontSize': '13px'})
                ], className='stat-card', style={
                    'background': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                    'color': 'white'
                })
            ], className='col-md-3'),
            
            html.Div([
                html.Div([
                    html.I(className='fas fa-chart-bar', 
                          style={'fontSize': '28px', 'color': '#f39c12'}),
                    html.H4('60', style={'margin': '10px 0 5px 0', 'fontSize': '32px'}),
                    html.P('ترکیب آزمایش', style={'margin': 0, 'color': '#7f8c8d', 'fontSize': '13px'})
                ], className='stat-card', style={
                    'background': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
                    'color': 'white'
                })
            ], className='col-md-3')
        ], className='row', style={'marginBottom': '30px'}),
        
        # نمودار اصلی: Heatmap
        html.Div([
            dcc.Graph(figure=fig_heatmap)
        ], className='chart-container', style={'marginBottom': '20px'}),
        
        # ردیف اول نمودارها
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_bar_comparison)
            ], className='col-md-6'),
            
            html.Div([
                dcc.Graph(figure=fig_line_trend)
            ], className='col-md-6')
        ], className='row', style={'marginBottom': '20px'}),
        
        # ردیف دوم نمودارها
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_box_collision)
            ], className='col-md-6'),
            
            html.Div([
                dcc.Graph(figure=fig_radar_layers)
            ], className='col-md-6')
        ], className='row', style={'marginBottom': '20px'}),
        
        # نمودار Scatter تمام عرض
        html.Div([
            dcc.Graph(figure=fig_scatter)
        ], className='chart-container', style={'marginBottom': '20px'}),
        
        # جدول خلاصه
        html.Div([
            html.H4('📋 جدول خلاصه نتایج',
                   style={'marginBottom': '15px', 'color': '#2c3e50'}),
            html.Div([
                html.Table([
                    # Header
                    html.Thead(html.Tr([
                        html.Th('سطح پیچیدگی', style={'backgroundColor': '#34495e', 'color': 'white'}),
                        html.Th('الگوریتم', style={'backgroundColor': '#34495e', 'color': 'white'}),
                        html.Th('لایه', style={'backgroundColor': '#34495e', 'color': 'white'}),
                        html.Th('تأخیر (ms)', style={'backgroundColor': '#34495e', 'color': 'white'}),
                        html.Th('انرژی (J)', style={'backgroundColor': '#34495e', 'color': 'white'}),
                        html.Th('موفقیت (%)', style={'backgroundColor': '#34495e', 'color': 'white'}),
                        html.Th('برخورد (%)', style={'backgroundColor': '#34495e', 'color': 'white'}),
                        html.Th('ایمنی (%)', style={'backgroundColor': '#34495e', 'color': 'white'})
                    ])),
                    
                    # Body (نمونه - 10 ردیف اول)
                    html.Tbody([
                        html.Tr([
                            html.Td(row['Complexity']),
                            html.Td(row['Algorithm']),
                            html.Td(row['Layer']),
                            html.Td(f"{row['Delay']:.1f}"),
                            html.Td(f"{row['Energy']:.1f}"),
                            html.Td(f"{row['Success_Rate']:.1f}"),
                            html.Td(f"{row['Collision_Rate']:.1f}"),
                            html.Td(f"{row['Safety_Score']:.1f}")
                        ]) for _, row in df.head(10).iterrows()
                    ])
                ], className='table table-striped table-hover',
                   style={'fontSize': '13px'})
            ], style={
                'maxHeight': '400px',
                'overflowY': 'auto',
                'border': '1px solid #ddd',
                'borderRadius': '5px'
            })
        ], className='table-container', style={'marginTop': '30px'})
        
    ], style={'padding': '20px'})
    
    return layout
