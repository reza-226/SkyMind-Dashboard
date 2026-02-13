import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os

# ==================== تنظیمات فونت فارسی (بزرگتر و ضخیم‌تر) ====================
FONT_CONFIG = {
    'family': 'B Nazanin, Tahoma, Arial',
    'size': 18,  # افزایش از 14 به 18
    'color': '#2C3E50'
}

TITLE_FONT = {
    'family': 'B Nazanin, Tahoma, Arial',
    'size': 24,  # افزایش از 18 به 24
    'color': '#1A252F'
}

AXIS_FONT = {
    'family': 'B Nazanin, Tahoma, Arial',
    'size': 17,  # افزایش از 13 به 17
    'color': '#34495E'
}

LEGEND_FONT = {
    'family': 'B Nazanin, Tahoma, Arial',
    'size': 16,  # افزایش از 12 به 16
    'color': '#2C3E50'
}

# تنظیمات فونت برای متن روی نمودارها
TEXT_FONT = {
    'family': 'B Nazanin, Tahoma, Arial',
    'size': 16,  # افزایش از 12-13 به 16
    'color': '#2C3E50'
}

LAYOUT_CONFIG = {
    'plot_bgcolor': '#F8F9FA',
    'paper_bgcolor': 'white',
    'font': FONT_CONFIG,
    'margin': dict(l=80, r=50, t=100, b=80),
    'hoverlabel': dict(
        bgcolor="white",
        font_size=16,  # افزایش از 13 به 16
        font_family="B Nazanin, Tahoma, Arial"
    )
}

# ==================== داده‌ها ====================
# جدول 5-1: مقایسه مصرف انرژی
energy_data = {
    'روش': ['MADDPG\n(پیشنهادی)', 'Random', 'Always Local', 'Always Edge', 'Round Robin', 'Load Balance'],
    'مصرف باتری': [3.82, 8.91, 9.45, 4.23, 5.67, 4.89],
    'کاهش نسبت به MADDPG': [0, 5.09, 5.63, 0.41, 1.85, 1.07],
    'کاهش درصدی': [0, 57, 59, 10, 32, 22]
}

# جدول 5-2: تحلیل اجزای تأخیر
latency_data = {
    'روش': ['MADDPG', 'Random', 'Always Local', 'Always Edge'],
    'تأخیر انتقال': [18.4, 35.2, 8.1, 28.6],
    'تأخیر صف': [12.6, 58.3, 2.4, 24.2],
    'تأخیر پردازش': [23.2, 32.1, 78.8, 25.7],
    'تأخیر کل': [54.2, 125.7, 89.3, 78.6]
}

# جدول 5-3: تأثیر پیچیدگی
complexity_data = {
    'سطح': ['آسان', 'متوسط', 'پیچیده'],
    'موانع': [0, 2, 4],
    'باتری': [3.82, 4.25, 4.89],
    'تأخیر': [54.2, 62.3, 71.1],
    'اشباع': [12, 15, 19],
    'موفقیت': [97, 95, 93]
}

# جدول 5-4: توزیع انتخاب لایه
layer_distribution = {
    'سطح': ['آسان', 'متوسط', 'پیچیده'],
    'Local': [28.3, 24.5, 18.2],
    'Terrestrial Edge': [52.3, 38.7, 20.4],
    'Aerial Edge': [12.8, 24.2, 55.8],
    'Cloud': [6.6, 12.6, 5.6]
}

# جدول 5-5: Ablation Study
ablation_data = {
    'واریانت': ['Full Model', 'No GAT', 'No Temporal\n(GRU)', 'Decentralized', 'Simpler Arch'],
    'Best Reward': [130.53, 95.24, 118.63, 65.81, 45.69],
    'Final Avg': [12.34, -20.24, -26.63, -85.81, -82.69],
    'Actor Loss': [2.87, 4.92, 5.82, 8.34, 12.45],
    'Critic Loss': [4.23, 8.45, 7.91, 15.67, 18.92]
}

# جدول 5-6: تحلیل آماری
statistical_data = {
    'واریانت': ['Full Model', 'No GAT', 'No Temporal', 'Decentralized', 'Simpler Arch'],
    "Cohen's d": [0.0, 0.3774, -0.0758, 0.4923, 1.1250],
    'p-value': [1.0, 8.57e-03, 5.94e-01, 6.52e-04, 1.72e-13],
    'معناداری': ['Baseline', 'معنی‌دار ⭐', 'ناچیز', 'بسیار معنی‌دار ⭐⭐', 'شدیداً معنی‌دار ⭐⭐⭐']
}

# ==================== توابع رسم نمودارها ====================

def create_energy_comparison():
    """نمودار 1: مقایسه کارایی انرژی (جدول 5-1)"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=energy_data['روش'],
        x=energy_data['مصرف باتری'],
        orientation='h',
        marker=dict(
            color=energy_data['مصرف باتری'],
            colorscale=[[0, '#27AE60'], [0.4, '#F39C12'], [1, '#E74C3C']],
            showscale=False,
            line=dict(color='white', width=2)
        ),
        text=[f"<b>{val:.2f} mJ</b><br>(-{perc}%)" if perc > 0 else f"<b>{val:.2f} mJ</b>" 
              for val, perc in zip(energy_data['مصرف باتری'], energy_data['کاهش درصدی'])],
        textposition='outside',
        textfont=dict(family='B Nazanin, Tahoma, Arial', size=16, color='#2C3E50'),
        hovertemplate='<b>%{y}</b><br>مصرف باتری: %{x:.2f} mJ<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>مقایسه کارایی انرژی روش‌های مختلف</b><br><sub style="font-size:18px">بر اساس جدول 5-1</sub>',
            font=TITLE_FONT,
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>مصرف باتری (mJ)</b>',
            titlefont=AXIS_FONT,
            tickfont=AXIS_FONT,
            gridcolor='#E0E0E0',
            gridwidth=1
        ),
        yaxis=dict(
            title='',
            tickfont=AXIS_FONT
        ),
        **LAYOUT_CONFIG,
        height=500,
        showlegend=False
    )
    
    return fig

def create_latency_total():
    """نمودار 2: مقایسه تأخیر کل"""
    
    fig = go.Figure()
    
    colors_map = {
        'MADDPG': '#27AE60',
        'Random': '#E74C3C',
        'Always Local': '#F39C12',
        'Always Edge': '#3498DB'
    }
    
    fig.add_trace(go.Bar(
        x=latency_data['روش'],
        y=latency_data['تأخیر کل'],
        marker=dict(
            color=[colors_map[m] for m in latency_data['روش']],
            line=dict(color='white', width=2)
        ),
        text=[f"<b>{val:.1f} ms</b>" for val in latency_data['تأخیر کل']],
        textposition='outside',
        textfont=dict(family='B Nazanin, Tahoma, Arial', size=16, color='#2C3E50'),
        hovertemplate='<b>%{x}</b><br>تأخیر کل: %{y:.1f} ms<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>مقایسه تأخیر کل در روش‌های مختلف</b><br><sub style="font-size:18px">بر اساس جدول 5-2</sub>',
            font=TITLE_FONT,
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>روش</b>',
            titlefont=AXIS_FONT,
            tickfont=AXIS_FONT
        ),
        yaxis=dict(
            title='<b>تأخیر کل (ms)</b>',
            titlefont=AXIS_FONT,
            tickfont=AXIS_FONT,
            gridcolor='#E0E0E0'
        ),
        **LAYOUT_CONFIG,
        height=500
    )
    
    return fig

def create_latency_breakdown():
    """نمودار 3: تجزیه اجزای تأخیر"""
    
    fig = go.Figure()
    
    components = {
        'تأخیر انتقال': ('#3498DB', latency_data['تأخیر انتقال']),
        'تأخیر صف': ('#F39C12', latency_data['تأخیر صف']),
        'تأخیر پردازش': ('#27AE60', latency_data['تأخیر پردازش'])
    }
    
    for comp_name, (color, values) in components.items():
        fig.add_trace(go.Bar(
            name=comp_name,
            x=latency_data['روش'],
            y=values,
            marker=dict(color=color, line=dict(color='white', width=1.5)),
            text=[f"<b>{v:.1f}</b>" for v in values],
            textposition='inside',
            textfont=dict(family='B Nazanin, Tahoma, Arial', size=15, color='white'),
            hovertemplate='<b>%{x}</b><br>' + comp_name + ': %{y:.1f} ms<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>تجزیه اجزای تأخیر</b><br><sub style="font-size:18px">بر اساس جدول 5-2</sub>',
            font=TITLE_FONT,
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>روش</b>',
            titlefont=AXIS_FONT,
            tickfont=AXIS_FONT
        ),
        yaxis=dict(
            title='<b>تأخیر (ms)</b>',
            titlefont=AXIS_FONT,
            tickfont=AXIS_FONT,
            gridcolor='#E0E0E0'
        ),
        barmode='stack',
        legend=dict(
            font=LEGEND_FONT,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        **LAYOUT_CONFIG,
        height=550
    )
    
    return fig

def create_success_saturation():
    """نمودار 4: نرخ موفقیت و اشباع در سطوح پیچیدگی"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>نرخ موفقیت (%)</b>', '<b>درصد اشباع (%)</b>'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=complexity_data['سطح'],
            y=complexity_data['موفقیت'],
            marker=dict(
                color=complexity_data['موفقیت'],
                colorscale=[[0, '#E74C3C'], [0.5, '#F39C12'], [1, '#27AE60']],
                showscale=False,
                line=dict(color='white', width=2)
            ),
            text=[f"<b>{v}%</b>" for v in complexity_data['موفقیت']],
            textposition='outside',
            textfont=dict(family='B Nazanin, Tahoma, Arial', size=17),
            name='موفقیت',
            hovertemplate='<b>%{x}</b><br>نرخ موفقیت: %{y}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=complexity_data['سطح'],
            y=complexity_data['اشباع'],
            marker=dict(
                color=complexity_data['اشباع'],
                colorscale=[[0, '#27AE60'], [0.5, '#F39C12'], [1, '#E74C3C']],
                showscale=False,
                line=dict(color='white', width=2)
            ),
            text=[f"<b>{v}%</b>" for v in complexity_data['اشباع']],
            textposition='outside',
            textfont=dict(family='B Nazanin, Tahoma, Arial', size=17),
            name='اشباع',
            hovertemplate='<b>%{x}</b><br>درصد اشباع: %{y}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='<b>سطح پیچیدگی</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, row=1, col=1)
    fig.update_xaxes(title_text='<b>سطح پیچیدگی</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, row=1, col=2)
    fig.update_yaxes(title_text='<b>درصد</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, gridcolor='#E0E0E0', row=1, col=1)
    fig.update_yaxes(title_text='<b>درصد</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, gridcolor='#E0E0E0', row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text='<b>تأثیر پیچیدگی محیط بر نرخ موفقیت و اشباع</b><br><sub style="font-size:18px">بر اساس جدول 5-3</sub>',
            font=TITLE_FONT,
            x=0.5,
            xanchor='center'
        ),
        **LAYOUT_CONFIG,
        height=500,
        showlegend=False
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(family='B Nazanin, Tahoma, Arial', size=19)
    
    return fig

def create_layer_distribution():
    """نمودار 5: توزیع انتخاب لایه در سطوح مختلف"""
    
    fig = go.Figure()
    
    layers = {
        'Local': '#3498DB',
        'Terrestrial Edge': '#27AE60',
        'Aerial Edge': '#F39C12',
        'Cloud': '#9B59B6'
    }
    
    for layer, color in layers.items():
        fig.add_trace(go.Bar(
            name=layer,
            x=layer_distribution['سطح'],
            y=layer_distribution[layer],
            marker=dict(color=color, line=dict(color='white', width=1.5)),
            text=[f"<b>{v:.1f}%</b>" for v in layer_distribution[layer]],
            textposition='inside',
            textfont=dict(family='B Nazanin, Tahoma, Arial', size=15, color='white'),
            hovertemplate='<b>%{x}</b><br>' + layer + ': %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>توزیع انتخاب لایه‌های پردازشی</b><br><sub style="font-size:18px">بر اساس جدول 5-4</sub>',
            font=TITLE_FONT,
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>سطح پیچیدگی</b>',
            titlefont=AXIS_FONT,
            tickfont=AXIS_FONT
        ),
        yaxis=dict(
            title='<b>درصد استفاده (%)</b>',
            titlefont=AXIS_FONT,
            tickfont=AXIS_FONT,
            gridcolor='#E0E0E0'
        ),
        barmode='stack',
        legend=dict(
            font=LEGEND_FONT,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        **LAYOUT_CONFIG,
        height=550
    )
    
    return fig

def create_complexity_effect():
    """نمودار 6: اثر پیچیدگی بر معیارهای عملکرد"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('<b>مصرف باتری</b>', '<b>تأخیر</b>', '<b>درصد اشباع</b>', '<b>نرخ موفقیت</b>'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    obstacles = complexity_data['موانع']
    
    fig.add_trace(
        go.Scatter(
            x=obstacles, y=complexity_data['باتری'],
            mode='lines+markers',
            line=dict(color='#E74C3C', width=4),
            marker=dict(size=12, color='#C0392B', line=dict(color='white', width=2)),
            name='باتری',
            hovertemplate='موانع: %{x}<br>باتری: %{y:.2f} mJ<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=obstacles, y=complexity_data['تأخیر'],
            mode='lines+markers',
            line=dict(color='#F39C12', width=4),
            marker=dict(size=12, color='#D68910', line=dict(color='white', width=2)),
            name='تأخیر',
            hovertemplate='موانع: %{x}<br>تأخیر: %{y:.1f} ms<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=obstacles, y=complexity_data['اشباع'],
            mode='lines+markers',
            line=dict(color='#9B59B6', width=4),
            marker=dict(size=12, color='#7D3C98', line=dict(color='white', width=2)),
            name='اشباع',
            hovertemplate='موانع: %{x}<br>اشباع: %{y}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=obstacles, y=complexity_data['موفقیت'],
            mode='lines+markers',
            line=dict(color='#27AE60', width=4),
            marker=dict(size=12, color='#1E8449', line=dict(color='white', width=2)),
            name='موفقیت',
            hovertemplate='موانع: %{x}<br>موفقیت: %{y}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text='<b>تعداد موانع</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, row=1, col=1)
    fig.update_xaxes(title_text='<b>تعداد موانع</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, row=1, col=2)
    fig.update_xaxes(title_text='<b>تعداد موانع</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, row=2, col=1)
    fig.update_xaxes(title_text='<b>تعداد موانع</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, row=2, col=2)
    
    fig.update_yaxes(title_text='<b>mJ</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, gridcolor='#E0E0E0', row=1, col=1)
    fig.update_yaxes(title_text='<b>ms</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, gridcolor='#E0E0E0', row=1, col=2)
    fig.update_yaxes(title_text='<b>%</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, gridcolor='#E0E0E0', row=2, col=1)
    fig.update_yaxes(title_text='<b>%</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, gridcolor='#E0E0E0', row=2, col=2)
    
    fig.update_layout(
        title=dict(
            text='<b>تأثیر پیچیدگی محیط بر معیارهای عملکرد</b><br><sub style="font-size:18px">بر اساس جدول 5-3</sub>',
            font=TITLE_FONT,
            x=0.5,
            xanchor='center'
        ),
        **LAYOUT_CONFIG,
        height=700,
        showlegend=False
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(family='B Nazanin, Tahoma, Arial', size=19)
    
    return fig

def create_strategy_change():
    """نمودار 7: تغییر استراتژی در سطوح مختلف"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=layer_distribution['سطح'],
        y=layer_distribution['Terrestrial Edge'],
        mode='lines+markers',
        name='Terrestrial Edge',
        line=dict(color='#27AE60', width=5),
        marker=dict(size=14, symbol='circle', line=dict(color='white', width=2)),
        fill='tonexty',
        fillcolor='rgba(39, 174, 96, 0.1)',
        hovertemplate='<b>%{x}</b><br>Terrestrial Edge: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=layer_distribution['سطح'],
        y=layer_distribution['Aerial Edge'],
        mode='lines+markers',
        name='Aerial Edge',
        line=dict(color='#F39C12', width=5),
        marker=dict(size=14, symbol='diamond', line=dict(color='white', width=2)),
        fill='tonexty',
        fillcolor='rgba(243, 156, 18, 0.1)',
        hovertemplate='<b>%{x}</b><br>Aerial Edge: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_annotation(
        x='پیچیده',
        y=55.8,
        text='<b>تغییر استراتژی:<br>55.8% استفاده از پهپادها</b>',
        showarrow=True,
        arrowhead=2,
        arrowcolor='#E74C3C',
        ax=-80,
        ay=-60,
        font=dict(family='B Nazanin, Tahoma, Arial', size=15, color='#E74C3C'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#E74C3C',
        borderwidth=2
    )
    
    fig.update_layout(
        title=dict(
            text='<b>تغییر استراتژی: از سرورهای زمینی به پهپادها</b><br><sub style="font-size:18px">بر اساس جدول 5-4</sub>',
            font=TITLE_FONT,
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='<b>سطح پیچیدگی</b>',
            titlefont=AXIS_FONT,
            tickfont=AXIS_FONT
        ),
        yaxis=dict(
            title='<b>درصد استفاده (%)</b>',
            titlefont=AXIS_FONT,
            tickfont=AXIS_FONT,
            gridcolor='#E0E0E0'
        ),
        legend=dict(
            font=LEGEND_FONT,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        **LAYOUT_CONFIG,
        height=550
    )
    
    return fig

def create_ablation_study():
    """نمودار 8: مطالعه حذفی (Ablation Study)"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>Best Reward</b>', '<b>Final Average (100 Last)</b>'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    colors = ['#27AE60', '#F39C12', '#3498DB', '#E74C3C', '#9B59B6']
    
    fig.add_trace(
        go.Bar(
            x=ablation_data['واریانت'],
            y=ablation_data['Best Reward'],
            marker=dict(color=colors, line=dict(color='white', width=2)),
            text=[f"<b>{v:.1f}</b>" for v in ablation_data['Best Reward']],
            textposition='outside',
            textfont=dict(family='B Nazanin, Tahoma, Arial', size=16),
            name='Best Reward',
            hovertemplate='<b>%{x}</b><br>Best Reward: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=ablation_data['واریانت'],
            y=ablation_data['Final Avg'],
            marker=dict(
                color=ablation_data['Final Avg'],
                colorscale=[[0, '#E74C3C'], [0.5, '#F39C12'], [1, '#27AE60']],
                showscale=False,
                line=dict(color='white', width=2)
            ),
            text=[f"<b>{v:.1f}</b>" for v in ablation_data['Final Avg']],
            textposition='outside',
            textfont=dict(family='B Nazanin, Tahoma, Arial', size=16),
            name='Final Avg',
            hovertemplate='<b>%{x}</b><br>Final Avg: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(tickangle=-15, titlefont=AXIS_FONT, tickfont=AXIS_FONT, row=1, col=1)
    fig.update_xaxes(tickangle=-15, titlefont=AXIS_FONT, tickfont=AXIS_FONT, row=1, col=2)
    fig.update_yaxes(title_text='<b>Reward</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, gridcolor='#E0E0E0', row=1, col=1)
    fig.update_yaxes(title_text='<b>Reward</b>', titlefont=AXIS_FONT, tickfont=AXIS_FONT, gridcolor='#E0E0E0', row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text='<b>مطالعه حذفی: مقایسه واریانت‌های مختلف</b><br><sub style="font-size:18px">بر اساس جدول 5-5</sub>',
            font=TITLE_FONT,
            x=0.5,
            xanchor='center'
        ),
        **LAYOUT_CONFIG,
        height=550,
        showlegend=False
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(family='B Nazanin, Tahoma, Arial', size=19)
    
    return fig

def create_radar_chart():
    """نمودار 9: نمودار راداری مقایسه کلی"""
    
    metrics = ['کاهش انرژی', 'کاهش تأخیر', 'نرخ موفقیت', 'پایداری', 'سازگاری']
    
    maddpg_scores = [100, 100, 97, 95, 93]
    baseline_scores = [42, 44, 45, 60, 40]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=maddpg_scores + [maddpg_scores[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        fillcolor='rgba(39, 174, 96, 0.3)',
        line=dict(color='#27AE60', width=4),
        marker=dict(size=10, color='#1E8449', line=dict(color='white', width=2)),
        name='MADDPG (پیشنهادی)',
        hovertemplate='<b>%{theta}</b><br>امتیاز: %{r}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=baseline_scores + [baseline_scores[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(color='#E74C3C', width=4, dash='dash'),
        marker=dict(size=10, color='#C0392B', line=dict(color='white', width=2)),
        name='Random (مبنا)',
        hovertemplate='<b>%{theta}</b><br>امتیاز: %{r}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>مقایسه جامع عملکرد: MADDPG vs Baseline</b><br><sub style="font-size:18px">تحلیل چندبعدی کلیه معیارها</sub>',
            font=TITLE_FONT,
            x=0.5,
            xanchor='center'
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(family='B Nazanin, Tahoma, Arial', size=16),
                gridcolor='#E0E0E0'
            ),
            angularaxis=dict(
                tickfont=dict(family='B Nazanin, Tahoma, Arial', size=17)
            )
        ),
        legend=dict(
            font=LEGEND_FONT,
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5
        ),
        **LAYOUT_CONFIG,
        height=600
    )
    
    return fig

# ==================== ذخیره‌سازی نمودارها ====================

def save_all_charts():
    """ذخیره تمام نمودارها در مسیر مشخص شده"""
    
    # مسیر ذخیره‌سازی
    output_dir = r'D:\Payannameh\SkyMind-Dashboard\thesis_charts'
    
    # ایجاد پوشه در صورت عدم وجود
    os.makedirs(output_dir, exist_ok=True)
    
    charts = {
        'slide16_energy_comparison.html': create_energy_comparison(),
        'slide17_latency_total.html': create_latency_total(),
        'slide17_latency_breakdown.html': create_latency_breakdown(),
        'slide18_success_saturation.html': create_success_saturation(),
        'slide19_layer_distribution.html': create_layer_distribution(),
        'slide20_complexity_effect.html': create_complexity_effect(),
        'slide21_strategy_change.html': create_strategy_change(),
        'slide22_ablation_study.html': create_ablation_study(),
        'slide23_radar_chart.html': create_radar_chart()
    }
    
    print("=" * 70)
    print("🚀 شروع تولید نمودارها با فونت‌های بزرگ‌تر و ضخیم‌تر")
    print(f"📁 مسیر ذخیره‌سازی: {output_dir}")
    print("=" * 70)
    print()
    
    for i, (filename, fig) in enumerate(charts.items(), 1):
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"✅ [{i}/9] {filename} ذخیره شد")
    
    print()
    print("=" * 70)
    print(f"🎉 {len(charts)} نمودار با موفقیت در مسیر زیر ذخیره شدند:")
    print(f"📂 {output_dir}")
    print()
    print("🎯 ویژگی‌های نمودارها:")
    print("   • فونت B Nazanin با سایز 16-24")
    print("   • عناوین ضخیم و برجسته")
    print("   • خوانایی بهینه برای ارائه")
    print("   • تعاملی با امکان hover و zoom")
    print("=" * 70)

# اجرای تولید نمودارها
if __name__ == "__main__":
    save_all_charts()
