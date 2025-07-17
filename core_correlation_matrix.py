import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
from io import BytesIO

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]

def extract_core_data(df):
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
    if time.iloc[0] > 1e6:
        time = time / 1000
    time = time - time.iloc[0]
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]
    return core_data

def generate_core_correlation_plot(core_df, filename):
    corr = core_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    formatted_corr = corr.applymap(lambda x: f"{x:.1g}" if pd.notnull(x) else "")
    sns.heatmap(corr, annot=formatted_corr, fmt="", cmap='coolwarm',
                vmin=-1, vmax=1, square=True, annot_kws={"size": 12},
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
    ax.set_title(f"Core Correlation Matrix\n{filename}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, corr

def build_dependency_graph(corr_matrix, threshold=0.9):
    G = nx.Graph()
    for core in corr_matrix.columns:
        G.add_node(core)
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
    return fig

def run_core_correlation_matrix():
    st.subheader("🧩 Core Correlation Matrix & Dependency Graph")

    uploaded_file = st.file_uploader("📂 Upload OCCT CSV File", type=["csv"], key="corr-upload")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            core_df = extract_core_data(df)

            # Correlation heatmap
            corr_fig, corr_matrix = generate_core_correlation_plot(core_df, uploaded_file.name)
            st.pyplot(corr_fig)

            # Download correlation matrix
            corr_buf = BytesIO()
            corr_fig.savefig(corr_buf, format="png")
            st.download_button(
                label="💾 Download Correlation Matrix as PNG",
                data=corr_buf.getvalue(),
                file_name=f"{uploaded_file.name.replace('.', '_')}_correlation_matrix.png",
                mime="image/png"
            )

            # Threshold for dependency graph
            corr_values = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).stack()
            min_corr = float(corr_values.min())
            threshold = st.slider("📏 Correlation Threshold for Dependency Graph", 
                                  min_value=round(min_corr,2), max_value=1.0, value=0.9,
                                  step=0.01)
            G = build_dependency_graph(corr_matrix, threshold=threshold)
            graph_fig = plot_dependency_graph(G, uploaded_file.name)
            st.pyplot(graph_fig)

            # Download dependency graph
            graph_buf = BytesIO()
            graph_fig.savefig(graph_buf, format='png')
            st.download_button(
                label="📥 Download Dependency Graph as PNG",
                data=graph_buf.getvalue(),
                file_name=f"{uploaded_file.name}_core_dependency_graph.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")