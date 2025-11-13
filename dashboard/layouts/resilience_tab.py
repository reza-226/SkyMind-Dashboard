from dash import dcc, html
import plotly.express as px

def layout(df_episodes):
    fig = px.line(df_episodes, x="Episode", y="Energy", title="Energy Resilience Over Episodes")
    return html.Div([
        html.H3("Resilience & Energy Stability", style={"textAlign": "center"}),
        dcc.Graph(figure=fig)
    ])
