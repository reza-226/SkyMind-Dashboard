# در فایل: pages/test_gnn_page.py

import streamlit as st
import torch
import plotly.graph_objects as go
import networkx as nx
from models.gnn.task_encoder import GNNTaskEncoder
from utils.graph_utils import TaskDAG, convert_dag_to_pyg_data

def show_gnn_test_page():
    st.title("🧪 GNN Task Encoder Test")
    
    # کنترل‌های تنظیمات
    col1, col2, col3 = st.columns(3)
    with col1:
        num_tasks = st.slider("تعداد Tasks", 5, 20, 10)
    with col2:
        num_deps = st.slider("تعداد وابستگی‌ها", 5, 30, 14)
    with col3:
        threshold = st.slider("آستانه Critical Path", 0.0, 1.0, 0.5)
    
    if st.button("🚀 اجرای تست", type="primary"):
        with st.spinner("در حال تولید DAG..."):
            # تولید DAG
            dag = generate_random_dag(num_tasks, num_deps)
            task_graph = convert_dag_to_pyg_data(dag)
            
            # ساخت encoder
            encoder = GNNTaskEncoder(
                node_feature_dim=9,
                edge_feature_dim=3,
                embedding_dim=256,
                num_gat_layers=3,
                num_heads=4
            )
            
            # Forward pass
            with torch.no_grad():
                embeddings, critical_scores = encoder(task_graph)
                critical_mask = encoder.get_critical_path(task_graph, threshold)
            
            # نمایش نتایج
            display_results(dag, embeddings, critical_scores, critical_mask)

def display_results(dag, embeddings, critical_scores, critical_mask):
    """نمایش نتایج تست"""
    
    # متریک‌های کلیدی
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("تعداد Tasks", dag.num_tasks)
    with col2:
        st.metric("تعداد وابستگی‌ها", len(dag.dependencies))
    with col3:
        critical_count = critical_mask.sum().item()
        st.metric("Tasks بحرانی", critical_count)
    with col4:
        avg_score = critical_scores.mean().item()
        st.metric("میانگین امتیاز", f"{avg_score:.3f}")
    
    # نمایش گراف تعاملی
    st.subheader("📊 نمایش گراف DAG")
    fig = plot_dag_with_scores(dag, critical_scores, critical_mask)
    st.plotly_chart(fig, use_container_width=True)
    
    # جدول اطلاعات tasks
    st.subheader("📋 اطلاعات Tasks")
    display_task_table(dag, critical_scores, critical_mask)
    
    # نمایش embeddings (t-SNE)
    st.subheader("🎯 نمایش Embeddings")
    fig_embedding = plot_embeddings_tsne(embeddings, critical_mask)
    st.plotly_chart(fig_embedding, use_container_width=True)

def plot_dag_with_scores(dag, scores, critical_mask):
    """رسم گراف DAG با امتیازات بحرانی"""
    
    G = nx.DiGraph()
    
    # اضافه کردن nodes
    for i in range(dag.num_tasks):
        G.add_node(i, 
                   score=scores[i].item(),
                   is_critical=bool(critical_mask[i]))
    
    # اضافه کردن edges
    for (src, dst) in dag.dependencies.keys():
        G.add_edge(src, dst)
    
    # محاسبه layout
    pos = nx.spring_layout(G, seed=42)
    
    # ساخت figure
    fig = go.Figure()
    
    # رسم edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # رسم nodes
    for node in G.nodes():
        x, y = pos[node]
        score = G.nodes[node]['score']
        is_critical = G.nodes[node]['is_critical']
        
        color = 'red' if is_critical else 'lightblue'
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(
                size=30,
                color=color,
                line=dict(width=2, color='darkblue')
            ),
            text=str(node),
            textposition='middle center',
            hovertemplate=f'Task {node}<br>Score: {score:.3f}<br>Critical: {is_critical}',
            showlegend=False
        ))
    
    fig.update_layout(
        title="DAG با مسیر بحرانی",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    return fig

def display_task_table(dag, scores, critical_mask):
    """نمایش جدول اطلاعات tasks"""
    import pandas as pd
    
    data = []
    for i in range(dag.num_tasks):
        task = dag.tasks[i]
        data.append({
            'Task ID': i,
            'Comp Demand': f"{task['comp_demand']:.2f}",
            'Data Size': f"{task['data_size']:.2f}",
            'Deadline': f"{task['deadline']:.2f}",
            'Critical Score': f"{scores[i].item():.3f}",
            'Is Critical': '✅' if critical_mask[i] else '❌'
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

def plot_embeddings_tsne(embeddings, critical_mask):
    """نمایش embeddings با t-SNE"""
    from sklearn.manifold import TSNE
    
    # اعمال t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.numpy())
    
    # ساخت figure
    fig = go.Figure()
    
    colors = ['red' if m else 'blue' for m in critical_mask]
    
    fig.add_trace(go.Scatter(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        mode='markers+text',
        marker=dict(size=15, color=colors),
        text=[str(i) for i in range(len(embeddings))],
        textposition='top center'
    ))
    
    fig.update_layout(
        title="نمایش Embeddings (t-SNE)",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=400
    )
    
    return fig
