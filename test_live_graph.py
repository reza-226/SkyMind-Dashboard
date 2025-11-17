# test_live_graph.py
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
from collections import deque
import numpy as np

MAX_POINTS = 30
data_store = {
    'x': deque(maxlen=MAX_POINTS),
    'y': deque(maxlen=MAX_POINTS),
    'counter': 0
}

app = Dash(__name__)

app.layout = html.Div([
    html.H1("TEST: Fixed Window Graph"),
    dcc.Graph(id='test-graph'),
    dcc.Interval(id='interval', interval=3000, n_intervals=0)
])

@app.callback(
    Output('test-graph', 'figure'),
    Input('interval', 'n_intervals')
)
def update(n):
    # اضافه کردن نقطه جدید
    data_store['counter'] += 1
    data_store['x'].append(data_store['counter'])
    data_store['y'].append(np.random.rand())
    
    # تبدیل به لیست
    x_data = list(data_store['x'])
    y_data = list(data_store['y'])
    
    # ساخت نمودار
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers'))
    
    # تنظیمات مهم
    if len(x_data) > 0:
        fig.update_layout(
            xaxis=dict(range=[min(x_data)-1, max(x_data)+1]),
            yaxis=dict(range=[0, 1]),
            uirevision='constant',
            transition={'duration': 0}
        )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8051)
