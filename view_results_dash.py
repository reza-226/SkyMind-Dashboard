# نام فایل: view_results_final.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pickle
import numpy as np

# =====================================================
# خواندن داده‌های کش
# =====================================================
print("📂 Loading cache data...")
with open('analysis/realtime/realtime_cache.pkl', 'rb') as f:
    data = pickle.load(f)

print("✅ Cache loaded successfully!")
print(f"📊 Episodes: {data['episodes']}")
print(f"⏱️  Duration: {data['duration_sec']:.2f}s")
print(f"📅 Timestamp: {data['timestamp']}")

# =====================================================
# استخراج داده‌های تاریخچه
# =====================================================
episodes = list(range(len(data['U_history'])))
U_history = data['U_history']
Delta_history = data['Delta_history']
Omega_history = data['Omega_history']
Energy_history = data['Energy_history']
Delay_history = data['Delay_history']

# محاسبه آمار
U_mean = data['mean_U']
Delta_mean = data['mean_Delta']
Omega_mean = data['mean_Omega']
Energy_mean = data['mean_Energy_J']
Delay_mean = data['mean_Delay_ms']
Energy_reduction = data['mean_Energy_Reduction_%']
Delay_reduction = data['mean_Delay_Reduction_%']

# =====================================================
# ساخت اپلیکیشن Dash
# =====================================================
app = dash.Dash(__name__)
app.title = "SkyMind Dashboard"

# =====================================================
# ساخت نمودارها
# =====================================================

# نمودار اصلی - 4 Subplots
fig_main = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        f'🎯 Utility (U) - Mean: {U_mean:.4f}',
        f'❌ Error (Δ) - Mean: {Delta_mean:.2f}%',
        f'⚡ Energy (J) - Reduction: {Energy_reduction:.1f}%',
        f'🚀 Delay (ms) - Reduction: {Delay_reduction:.1f}%'
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.10
)

# Utility
fig_main.add_trace(
    go.Scatter(
        x=episodes, y=U_history,
        name='Utility',
        line=dict(color='#2E86AB', width=2),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 171, 0.2)'
    ),
    row=1, col=1
)

# Error
fig_main.add_trace(
    go.Scatter(
        x=episodes, y=Delta_history,
        name='Error',
        line=dict(color='#C73E1D', width=2),
        fill='tozeroy',
        fillcolor='rgba(199, 62, 29, 0.2)'
    ),
    row=1, col=2
)

# Energy
fig_main.add_trace(
    go.Scatter(
        x=episodes, y=Energy_history,
        name='Energy',
        line=dict(color='#A23B72', width=2),
        fill='tozeroy',
        fillcolor='rgba(162, 59, 114, 0.2)'
    ),
    row=2, col=1
)

# Delay
fig_main.add_trace(
    go.Scatter(
        x=episodes, y=Delay_history,
        name='Delay',
        line=dict(color='#F18F01', width=2),
        fill='tozeroy',
        fillcolor='rgba(241, 143, 1, 0.2)'
    ),
    row=2, col=2
)

# بروزرسانی محورها
for i in range(1, 3):
    for j in range(1, 3):
        fig_main.update_xaxes(title_text="Episode", row=i, col=j)

fig_main.update_yaxes(title_text="Utility", row=1, col=1)
fig_main.update_yaxes(title_text="Error (%)", row=1, col=2)
fig_main.update_yaxes(title_text="Energy (J)", row=2, col=1)
fig_main.update_yaxes(title_text="Delay (ms)", row=2, col=2)

fig_main.update_layout(
    height=700,
    showlegend=False,
    title_text="📊 SkyMind Training Metrics - Complete History",
    font=dict(size=11),
    hovermode='x unified'
)

# نمودار Stability (Omega)
fig_omega = go.Figure()
fig_omega.add_trace(
    go.Scatter(
        x=episodes,
        y=Omega_history,
        name='Stability (Ω)',
        line=dict(color='#06A77D', width=3),
        fill='tozeroy',
        fillcolor='rgba(6, 167, 125, 0.2)'
    )
)
fig_omega.update_layout(
    title=f'🔒 System Stability (Ω) - Mean: {Omega_mean:.4f}',
    xaxis_title='Episode',
    yaxis_title='Stability (Ω)',
    height=400,
    hovermode='x'
)

# =====================================================
# Layout اپلیکیشن
# =====================================================
app.layout = html.Div([
    # Header
    html.Div([
        html.H1(
            '🚀 SkyMind Realtime Dashboard',
            style={
                'textAlign': 'center',
                'color': '#2E86AB',
                'marginBottom': '10px',
                'fontFamily': 'Arial, sans-serif'
            }
        ),
        html.P(
            f'📅 Training Session: {data["timestamp"]} | ⏱️ Duration: {data["duration_sec"]:.2f}s | 📊 Episodes: {data["episodes"]}',
            style={'textAlign': 'center', 'color': 'gray', 'fontSize': 14}
        ),
        html.Hr(style={'borderTop': '2px solid #2E86AB'})
    ]),
    
    # Metrics Cards
    html.Div([
        # Card 1: Utility
        html.Div([
            html.H3('🎯 Utility (U)', style={'color': '#2E86AB', 'marginBottom': '10px'}),
            html.H2(f'{U_mean:.4f}', style={'color': '#2E86AB', 'marginTop': '0'}),
            html.P(
                f'Min: {min(U_history):.4f} | Max: {max(U_history):.4f}',
                style={'fontSize': 12, 'color': 'gray'}
            ),
            html.P(
                f'📈 Improvement: {((U_history[-1]-U_history[0])/U_history[0]*100):.1f}%',
                style={'fontSize': 14, 'fontWeight': 'bold', 'color': 'green'}
            )
        ], style={
            'flex': '1',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '10px',
            'margin': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
        
        # Card 2: Error
        html.Div([
            html.H3('❌ Error (Δ)', style={'color': '#C73E1D', 'marginBottom': '10px'}),
            html.H2(f'{Delta_mean:.2f}%', style={'color': '#C73E1D', 'marginTop': '0'}),
            html.P(
                f'Min: {min(Delta_history):.2f}% | Max: {max(Delta_history):.2f}%',
                style={'fontSize': 12, 'color': 'gray'}
            ),
            html.P(
                f'📉 Reduction: {((Delta_history[0]-Delta_history[-1])/Delta_history[0]*100):.1f}%',
                style={'fontSize': 14, 'fontWeight': 'bold', 'color': 'green'}
            )
        ], style={
            'flex': '1',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '10px',
            'margin': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
        
        # Card 3: Energy
        html.Div([
            html.H3('⚡ Energy', style={'color': '#A23B72', 'marginBottom': '10px'}),
            html.H2(f'{Energy_mean:.3f} J', style={'color': '#A23B72', 'marginTop': '0'}),
            html.P(
                f'Min: {min(Energy_history):.3f}J | Max: {max(Energy_history):.3f}J',
                style={'fontSize': 12, 'color': 'gray'}
            ),
            html.P(
                f'⚡ Reduction: {Energy_reduction:.1f}%',
                style={'fontSize': 14, 'fontWeight': 'bold', 'color': 'green'}
            )
        ], style={
            'flex': '1',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '10px',
            'margin': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
        
        # Card 4: Delay
        html.Div([
            html.H3('🚀 Delay', style={'color': '#F18F01', 'marginBottom': '10px'}),
            html.H2(f'{Delay_mean:.1f} ms', style={'color': '#F18F01', 'marginTop': '0'}),
            html.P(
                f'Min: {min(Delay_history):.1f}ms | Max: {max(Delay_history):.1f}ms',
                style={'fontSize': 12, 'color': 'gray'}
            ),
            html.P(
                f'🚀 Reduction: {Delay_reduction:.1f}%',
                style={'fontSize': 14, 'fontWeight': 'bold', 'color': 'green'}
            )
        ], style={
            'flex': '1',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '10px',
            'margin': '10px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    
    # Main Charts
    dcc.Graph(figure=fig_main, style={'marginTop': '20px'}),
    
    # Stability Chart
    dcc.Graph(figure=fig_omega, style={'marginTop': '20px'}),
    
    # Summary Statistics
    html.Div([
        html.H3('📊 Summary Statistics:', style={'color': '#2E86AB', 'marginBottom': '15px'}),
        html.Div([
            html.P(f'✅ Average Utility: {U_mean:.4f}', style={'fontSize': 16}),
            html.P(f'❌ Average Error: {Delta_mean:.2f}%', style={'fontSize': 16}),
            html.P(f'🔒 Average Stability: {Omega_mean:.4f}', style={'fontSize': 16}),
            html.P(f'⚡ Energy Reduction: {Energy_reduction:.1f}%', style={'fontSize': 16, 'color': 'green', 'fontWeight': 'bold'}),
            html.P(f'🚀 Delay Reduction: {Delay_reduction:.1f}%', style={'fontSize': 16, 'color': 'green', 'fontWeight': 'bold'}),
        ])
    ], style={
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '10px',
        'margin': '20px 10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Footer
    html.Footer([
        html.Hr(style={'borderTop': '2px solid #2E86AB'}),
        html.P(
            '🔬 SkyMind Dashboard v1.0 | Realtime Analysis Module',
            style={'textAlign': 'center', 'color': 'gray', 'fontSize': 12}
        )
    ])
], style={
    'fontFamily': 'Arial, sans-serif',
    'padding': '20px',
    'backgroundColor': '#ffffff'
})

# =====================================================
# اجرای سرور
# =====================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎯 SkyMind Dashboard is running!")
    print("📡 Access at: http://127.0.0.1:8051")
    print("="*70 + "\n")
    print("📊 Loaded Data Summary:")
    print(f"   • Episodes: {data['episodes']}")
    print(f"   • Mean Utility: {U_mean:.4f}")
    print(f"   • Mean Error: {Delta_mean:.2f}%")
    print(f"   • Energy Reduction: {Energy_reduction:.1f}%")
    print(f"   • Delay Reduction: {Delay_reduction:.1f}%")
    print("="*70 + "\n")
    
    # ✅ اصلاح شده:
    app.run(debug=True, port=8051)
