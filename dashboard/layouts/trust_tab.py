def layout(df_trust):
    fig = px.line(df_trust, x="Episode", y="Trust", title="Trust Stability")
    return html.Div([dcc.Graph(figure=fig)])
