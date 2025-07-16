import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]

def extract_core_data(df):
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]
    return core_data

def build_dependency_graph(core_df, threshold=0.9):
    corr_matrix = core_df.corr()
    G = nx.Graph()

    # Add nodes
    for core in corr_matrix.columns:
        G.add_node(core)

    # Add edges where correlation > threshold
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            c1, c2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if corr_val >= threshold:
                G.add_edge(c1, c2, weight=round(corr_val, 2))

    return G

def plot_dependency_graph(G, filename):
    pos = nx.spring_layout(G, seed=42)
    weights = nx.get_edge_attributes(G, 'weight')

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1000, font_size=10, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_color='red', font_size=8, ax=ax)
    ax.set_title(f"🔗 Core Dependency Graph: {filename}")
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    st.download_button(
        label="📥 Download Graph as PNG",
        data=buf.getvalue(),
        file_name=f"{filename}_core_dependency_graph.png",
        mime="image/png"
    )

def run_core_dependency_graph():
    st.header("🔗 Core Dependency Graph")

    uploaded_file = st.file_uploader("📂 Upload OCCT CSV File", type=["csv"], key="dependency-upload")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            core_df = extract_core_data(df)

            threshold = st.slider("📏 Correlation Threshold", 0.7, 1.0, 0.9, step=0.01)
            G = build_dependency_graph(core_df, threshold=threshold)
            plot_dependency_graph(G, uploaded_file.name)

        except Exception as e:
            st.error(f"❌ Error: {e}")