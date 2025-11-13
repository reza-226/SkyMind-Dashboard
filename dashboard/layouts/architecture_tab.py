# architecture_tab.py
from dash import html, dcc
import plotly.graph_objects as go

def layout(df_queues):
    fig = go.Figure(go.Scatter(x=[0,1,2], y=[0,1,0], name="Topology"))
    fig.update_layout(title="Architecture Connectivity")
    return html.Div([
        html.H3("System Architecture Topology"),
        dcc.Graph(figure=fig)
    ])
