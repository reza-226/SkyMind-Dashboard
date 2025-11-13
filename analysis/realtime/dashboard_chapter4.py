"""
داشبورد تعاملی حرفه‌ای فصل ۴ - تحلیل عملکرد الگوریتم‌های بهینه‌سازی
نویسنده: تیم SkyMind
تاریخ: ۱۴۰۴/۰۸/۲۰
"""

import pickle
import webbrowser
from pathlib import Path
from threading import Timer

import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# تنظیمات مسیر (خودکار)
# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_PATH = SCRIPT_DIR / "realtime_cache.pkl"

# ═══════════════════════════════════════════════════════════════════════
# بارگذاری داده‌ها
# ═══════════════════════════════════════════════════════════════════════
def load_realtime_cache():
    """بارگذاری cache با مدیریت خطا"""
    try:
        with open(CACHE_PATH, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"❌ خطا: فایل {CACHE_PATH} یافت نشد!")
        return None
    except Exception as e:
        print(f"❌ خطا در بارگذاری: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════
# ایجاد اپلیکیشن Dash با فعال‌سازی assets
# ═══════════════════════════════════════════════════════════════════════
app = Dash(__name__, assets_folder='assets')
app.title = "🚀 SkyMind Analytics Dashboard"

# ═══════════════════════════════════════════════════════════════════════
# طراحی Layout حرفه‌ای
# ═══════════════════════════════════════════════════════════════════════
app.layout = html.Div([
    # هدر با گرادیانت
    html.Div([
        html.H1("🚀 داشبورد تحلیل هوشمند SkyMind", 
                style={
                    'textAlign': 'center',
                    'margin': 0,
                    'fontSize': '2.5rem',
                    'fontWeight': '700'
                }),
        html.P("فصل ۴: مقایسه عملکرد الگوریتم‌های بهینه‌سازی چندهدفه", 
               style={
                   'textAlign': 'center',
                   'fontSize': '1.1rem',
                   'marginTop': '10px',
                   'opacity': '0.9'
               })
    ], className='main-header animated'),
    
    # پنل کنترل
    html.Div([
        html.Div([
            html.Label("🎯 انتخاب الگوریتم:", 
                      style={
                          'fontWeight': 'bold',
                          'fontSize': '1.1rem',
                          'marginBottom': '10px',
                          'color': '#ecf0f1'
                      }),
            dcc.Dropdown(
                id='algorithm-selector',
                options=[
                    {'label': '🐌 Greedy (پایه)', 'value': 'greedy'},
                    {'label': '🧬 GA (ژنتیک)', 'value': 'ga'},
                    {'label': '🔍 BLS (جستجوی محلی)', 'value': 'bls'},
                    {'label': '🤖 DDQN (یادگیری تقویتی)', 'value': 'ddqn'},
                    {'label': '⚡ ECORI (تعاونی)', 'value': 'ecori'},
                    {'label': '🏆 MADDPG (پیشنهادی)', 'value': 'maddpg'}
                ],
                value='maddpg',
                style={'width': '100%'},
                className='animated'
            )
        ], style={'maxWidth': '400px', 'margin': '0 auto'})
    ], className='card animated', style={'marginBottom': '20px'}),
    
    # تب‌های اصلی
    dcc.Tabs(id='tabs', value='tab-convergence', 
             style={'marginBottom': '20px'},
             children=[
        dcc.Tab(label='📈 تحلیل همگرایی', value='tab-convergence',
                className='tab'),
        dcc.Tab(label='⚡ عملکرد سیستم', value='tab-performance',
                className='tab'),
        dcc.Tab(label='🎯 جبهه پارتو', value='tab-pareto',
                className='tab'),
        dcc.Tab(label='📊 تحلیل آماری', value='tab-statistics',
                className='tab')
    ]),
    
    # محتوای اصلی
    html.Div(id='tab-content', className='animated')
    
], style={
    'maxWidth': '1400px',
    'margin': '0 auto',
    'padding': '30px',
    'minHeight': '100vh'
})

# ═══════════════════════════════════════════════════════════════════════
# Callback برای تغییر محتوای تب‌ها
# ═══════════════════════════════════════════════════════════════════════
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value'),
     Input('algorithm-selector', 'value')]
)
def render_content(tab, algorithm):
    data = load_realtime_cache()
    if not data or algorithm not in data:
        return html.Div([
            html.H3("⚠️ داده‌ای برای نمایش موجود نیست", 
                   style={'textAlign': 'center', 'color': '#e74c3c', 'marginTop': '50px'}),
            html.P("لطفاً ابتدا create_realtime_cache.py را اجرا کنید",
                  style={'textAlign': 'center', 'color': '#95a5a6'})
        ], className='card')
    
    algo_data = data[algorithm]
    
    if tab == 'tab-convergence':
        return create_convergence_tab(algo_data, algorithm)
    elif tab == 'tab-performance':
        return create_performance_tab(algo_data, algorithm)
    elif tab == 'tab-pareto':
        return create_pareto_tab(data)
    elif tab == 'tab-statistics':
        return create_statistics_tab(data)

# ═══════════════════════════════════════════════════════════════════════
# تابع ۱: تب همگرایی
# ═══════════════════════════════════════════════════════════════════════
def create_convergence_tab(algo_data, algo_name):
    """نمودار همگرایی utility با طراحی حرفه‌ای"""
    try:
        utility = algo_data.get('mean_TotalUtility', [])
        episodes = list(range(1, len(utility) + 1))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=episodes, y=utility,
            mode='lines+markers',
            name=algo_name.upper(),
            line=dict(color='#00d4ff', width=3, shape='spline'),
            marker=dict(size=8, color='#00d4ff', 
                       line=dict(width=2, color='#ffffff')),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ))
        
        fig.update_layout(
            title={
                'text': f'🎯 نمودار همگرایی Total Utility<br><sub>{algo_name.upper()}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#ecf0f1'}
            },
            xaxis_title='تعداد Episode',
            yaxis_title='مقدار Utility',
            hovermode='x unified',
            template='plotly_dark',
            height=550,
            plot_bgcolor='rgba(0, 0, 0, 0.3)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family='IRANSans, Arial', size=14)
        )
        
        # محاسبه آمار
        final_avg = np.mean(utility[-10:])
        max_val = np.max(utility)
        improvement = ((utility[-1] - utility[0]) / utility[0] * 100) if utility[0] != 0 else 0
        
        return html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': False}),
            
            # کارت‌های آماری
            html.Div([
                html.Div([
                    html.H4("📊 میانگین ۱۰ اپیزود آخر"),
                    html.H2(f"{final_avg:.2f}", style={'color': '#00d4ff'})
                ], className='card', style={'flex': '1', 'margin': '10px', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4("🔝 بیشترین مقدار"),
                    html.H2(f"{max_val:.2f}", style={'color': '#2ecc71'})
                ], className='card', style={'flex': '1', 'margin': '10px', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4("📈 میزان بهبود"),
                    html.H2(f"{improvement:+.1f}%", style={'color': '#f39c12'})
                ], className='card', style={'flex': '1', 'margin': '10px', 'textAlign': 'center'})
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': '20px'})
        ], className='card')
        
    except Exception as e:
        return html.Div(f"❌ خطا در رسم نمودار: {e}", 
                       style={'color': '#e74c3c', 'textAlign': 'center'})

# ═══════════════════════════════════════════════════════════════════════
# تابع ۲: تب عملکرد
# ═══════════════════════════════════════════════════════════════════════
def create_performance_tab(algo_data, algo_name):
    """نمودار انرژی و تأخیر با طراحی دوگانه"""
    try:
        energy = algo_data.get('mean_Energy_J', [])
        delay = algo_data.get('mean_Delay_ms', [])
        episodes = list(range(1, len(energy) + 1))
        
        fig = go.Figure()
        
        # نمودار انرژی
        fig.add_trace(go.Scatter(
            x=episodes, y=energy,
            mode='lines',
            name='مصرف انرژی (J)',
            yaxis='y1',
            line=dict(color='#ff6b6b', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        
        # نمودار تأخیر
        fig.add_trace(go.Scatter(
            x=episodes, y=delay,
            mode='lines',
            name='تأخیر میانگین (ms)',
            yaxis='y2',
            line=dict(color='#4ecdc4', width=3),
            fill='tozeroy',
            fillcolor='rgba(78, 205, 196, 0.1)'
        ))
        
        fig.update_layout(
            title={
                'text': f'⚡ تحلیل انرژی و تأخیر<br><sub>{algo_name.upper()}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#ecf0f1'}
            },
            xaxis_title='Episode',
            yaxis=dict(
                title='مصرف انرژی (J)',
                side='left',
                color='#ff6b6b'
            ),
            yaxis2=dict(
                title='تأخیر (ms)',
                overlaying='y',
                side='right',
                color='#4ecdc4'
            ),
            hovermode='x unified',
            template='plotly_dark',
            height=550,
            plot_bgcolor='rgba(0, 0, 0, 0.3)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family='IRANSans, Arial', size=14)
        )
        
        return html.Div(dcc.Graph(figure=fig, config={'displayModeBar': False}), 
                       className='card')
        
    except Exception as e:
        return html.Div(f"❌ خطا: {e}", style={'color': '#e74c3c'})

# ═══════════════════════════════════════════════════════════════════════
# تابع ۳: تب پارتو
# ═══════════════════════════════════════════════════════════════════════
def create_pareto_tab(data):
    """نمودار Pareto Front سه‌بعدی"""
    try:
        fig = go.Figure()
        
        colors = {
            'greedy': '#95a5a6',
            'ga': '#e67e22',
            'bls': '#9b59b6',
            'ddqn': '#34495e',
            'ecori': '#16a085',
            'maddpg': '#e74c3c'
        }
        
        for algo_name, algo_data in data.items():
            energy = np.mean(algo_data.get('mean_Energy_J', [0]))
            delay = np.mean(algo_data.get('mean_Delay_ms', [0]))
            
            is_proposed = (algo_name == 'maddpg')
            
            fig.add_trace(go.Scatter(
                x=[energy],
                y=[delay],
                mode='markers+text',
                name=algo_name.upper(),
                marker=dict(
                    size=25 if is_proposed else 18,
                    color=colors.get(algo_name, '#000'),
                    line=dict(width=3 if is_proposed else 1, 
                             color='#ffffff')
                ),
                text=[algo_name.upper()],
                textposition='top center',
                textfont=dict(size=14, color='#ffffff')
            ))
        
        fig.update_layout(
            title={
                'text': '🎯 مقایسه الگوریتم‌ها در فضای Energy-Delay<br><sub>هر چه نزدیک‌تر به مبدأ، بهتر</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#ecf0f1'}
            },
            xaxis_title='مصرف انرژی (J)',
            yaxis_title='تأخیر میانگین (ms)',
            template='plotly_dark',
            height=650,
            plot_bgcolor='rgba(0, 0, 0, 0.3)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            showlegend=True,
            legend=dict(
                bgcolor='rgba(44, 62, 80, 0.8)',
                bordercolor='#3498db',
                borderwidth=2
            ),
            font=dict(family='IRANSans, Arial', size=14)
        )
        
        return html.Div(dcc.Graph(figure=fig, config={'displayModeBar': False}), 
                       className='card')
        
    except Exception as e:
        return html.Div(f"❌ خطا: {e}", style={'color': '#e74c3c'})

# ═══════════════════════════════════════════════════════════════════════
# تابع ۴: تب آماری
# ═══════════════════════════════════════════════════════════════════════
def create_statistics_tab(data):
    """جدول آماری با طراحی مدرن"""
    try:
        stats = []
        for algo_name, algo_data in data.items():
            stats.append({
                'الگوریتم': algo_name.upper(),
                'انرژی (J)': f"{np.mean(algo_data.get('mean_Energy_J', [0])):.4f}",
                'تأخیر (ms)': f"{np.mean(algo_data.get('mean_Delay_ms', [0])):.2f}",
                'Utility': f"{np.mean(algo_data.get('mean_TotalUtility', [0])):.2f}",
                'انحراف معیار': f"{np.std(algo_data.get('mean_TotalUtility', [0])):.2f}"
            })
        
        df = pd.DataFrame(stats)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='#667eea',
                align='center',
                font=dict(color='white', size=16, family='IRANSans, Arial'),
                height=40
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color=[['#2c3e50', '#34495e'] * len(df)],
                align='center',
                font=dict(size=14, color='#ecf0f1', family='IRANSans, Arial'),
                height=35
            )
        )])
        
        fig.update_layout(
            title={
                'text': '📊 جدول مقایسه‌ای تفصیلی عملکرد',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#ecf0f1'}
            },
            height=450,
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(family='IRANSans, Arial')
        )
        
        return html.Div(dcc.Graph(figure=fig, config={'displayModeBar': False}), 
                       className='card')
        
    except Exception as e:
        return html.Div(f"❌ خطا: {e}", style={'color': '#e74c3c'})

# ═══════════════════════════════════════════════════════════════════════
# تابع باز کردن مرورگر
# ═══════════════════════════════════════════════════════════════════════
def open_browser():
    """باز کردن خودکار مرورگر"""
    webbrowser.open_new('http://127.0.0.1:8050/')

# ═══════════════════════════════════════════════════════════════════════
# اجرای برنامه
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 داشبورد تحلیل هوشمند SkyMind - نسخه حرفه‌ای")
    print("="*70)
    print(f"📂 مسیر داده‌ها: {CACHE_PATH}")
    
    if CACHE_PATH.exists():
        print("✅ فایل cache موجود است")
        data = load_realtime_cache()
        if data:
            print(f"📊 تعداد الگوریتم‌ها: {len(data)}")
            print(f"🌐 لینک داشبورد: http://127.0.0.1:8050")
            print(f"🎨 تم: Dark Mode با فونت فارسی")
            print("🔄 مرورگر در ۲ ثانیه باز می‌شود...")
            print("="*70 + "\n")
            
            # باز کردن مرورگر
            Timer(2.0, open_browser).start()
            
            app.run(debug=True, port=8050)
        else:
            print("❌ خطا در بارگذاری داده‌ها")
    else:
        print(f"❌ فایل cache یافت نشد!")
        print("💡 لطفاً ابتدا create_realtime_cache.py را اجرا کنید")
