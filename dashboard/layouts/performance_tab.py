from dash import html, dcc
import plotly.express as px

def layout(df_summary):
    if df_summary.empty:
        fig = px.line(x=[0,1], y=[0,1], title="No Performance Data Loaded")
    else:
        fig = px.line(df_summary, x="Episode", y="Utility", title="Performance – Utility Curve")
    return html.Div([
        html.H3("Performance Overview", style={"textAlign": "center"}),
        dcc.Graph(figure=fig)
    ])
