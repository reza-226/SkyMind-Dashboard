# app.py - فقط تغییرات لازم

# در بخش import ها:
from dashboard.layouts import (
    performance_tab,
    architecture_tab,
    resilience_tab,
    summary_tab,
    trust_tab,
    multi_env_tab  # ✅ اضافه شد
)

# در بخش Tabs (خط 58):
dcc.Tabs(id="tabs", value="performance", children=[
    dcc.Tab(label='Performance', value='performance'),
    dcc.Tab(label='Architecture', value='architecture'),
    dcc.Tab(label='Trust', value='trust'),
    dcc.Tab(label='Resilience', value='resilience'),
    dcc.Tab(label='Summary', value='summary'),
    dcc.Tab(label='🌐 Multi-Env', value='multi_env'),  # ✅ اضافه شد
]),

# در تابع render_content (خط 74):
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
    elif tab == "multi_env":  # ✅ اضافه شد
        return multi_env_tab.layout()
    else:
        return html.Div("Unknown tab")
