# ---------------------------------------------------------
# File: dashboard/app.py
# Description: SkyMind Scientific Dashboard – v6.4 Final UI
# Grounded on structure found in project_structure.txt (165k lines)
# ---------------------------------------------------------

from dash import Dash, dcc, html
from flask import Flask
from flask_socketio import SocketIO
import pandas as pd
import plotly.express as px

# مسیرهای داخلی بر اساس ساختار واقعی پروژه
from dashboard.layouts import (
    performance_tab,
    architecture_tab,
    resilience_tab,
    summary_tab,
    trust_tab
)
from dashboard.components import (
    kpi_charts,
    trust_monitor,
    arch3d,
    queue_chart
)

# ---------------------------------------------------------
# 1️⃣ ایجاد سرور Flask + Dash
server = Flask(__name__)
app = Dash(__name__, server=server, suppress_callback_exceptions=True)
socketio = SocketIO(server, cors_allowed_origins="*")

app.title = "SkyMind Scientific Dashboard"

# ---------------------------------------------------------
# 2️⃣ آماده‌سازی داده‌ها از ./data
try:
    df_summary = pd.read_csv("data/summary.csv")
    df_trust = pd.read_csv("data/trust.csv")
    df_episodes = pd.read_csv("data/episodes.csv")
    df_queues = pd.read_csv("data/queues.csv")
except Exception as e:
    print(f"[Warning] Could not load data files: {e}")
    df_summary = pd.DataFrame()
    df_trust = pd.DataFrame()
    df_episodes = pd.DataFrame()
    df_queues = pd.DataFrame()

# ---------------------------------------------------------
# 3️⃣ بخش بالایی داشبورد (Header + Tabs)
app.layout = html.Div([
    html.H1("🧠 SkyMind Scientific Dashboard", style={
        "textAlign": "center",
        "backgroundColor": "#222",
        "color": "white",
        "padding": "15px",
        "borderRadius": "8px"
    }),
    dcc.Tabs(id="tabs", value="performance", children=[
        dcc.Tab(label='Performance', value='performance'),
        dcc.Tab(label='Architecture', value='architecture'),
        dcc.Tab(label='Trust', value='trust'),
        dcc.Tab(label='Resilience', value='resilience'),
        dcc.Tab(label='Summary', value='summary'),
    ]),
    html.Div(id="tab-content", style={"margin": "20px"})
])

# ---------------------------------------------------------
# 4️⃣ کال‌بک برای تعویض تب‌ها
@app.callback(
    dash.dependencies.Output("tab-content", "children"),
    [dash.dependencies.Input("tabs", "value")]
)
def render_content(tab):
    if tab == "performance":
        return performance_tab.layout(df_summary)
    elif tab == "architecture":
        return architecture_tab.layout(df_queues)
    elif tab == "trust":
        return trust_tab.layout(df_trust)
    elif tab == "resilience":
        return resilience_tab.layout(df_episodes)
    elif tab == "summary":
        return summary_tab.layout(df_summary, df_trust)
    else:
        return html.Div("Unknown tab")

# ---------------------------------------------------------
# 5️⃣ توابع ارسال داده‌های زنده (SocketIO)
@socketio.on('connect')
def handle_connect():
    print("Client connected to SkyMind Dashboard")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

# ---------------------------------------------------------
# 6️⃣ اجرای نهایی سرور ‑ پشتیبانی از Flask و Dash
if __name__ == "__main__":
    print("🚀 Starting SkyMind Scientific Dashboard Server ...")
    print("🔗 Visit: http://127.0.0.1:8050/")
    socketio.run(server, host="0.0.0.0", port=8050, debug=True)
