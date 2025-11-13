from dash import html, dcc
import plotly.express as px

def layout(df_summary, df_trust):
    if df_summary.empty or df_trust.empty:
        return html.Div([
            html.H3("Summary Report: No scientific data loaded"),
            html.P("Run scripts/generate_dashboard_data.py to restore equilibrium data.")
        ])

    # نمودار ترکیبی Utility و Trust
    fig1 = px.line(df_summary, x="Episode", y="Utility", title="Scientific Utility (U)")
    fig2 = px.line(df_trust, x="Episode", y="Trust", title="Trust Stability")

    return html.Div([
        html.H2("Summary & Trust Overview", style={"textAlign": "center"}),
        html.Div([
            dcc.Graph(figure=fig1, style={"width": "48%", "display": "inline-block"}),
            dcc.Graph(figure=fig2, style={"width": "48%", "display": "inline-block"})
        ])
    ])
