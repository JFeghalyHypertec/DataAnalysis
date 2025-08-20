import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
from io import BytesIO
import plotly.graph_objects as go
import plotly.io as pio

# -----------------------------
# Constants
# -----------------------------
START_ROW = 3  # Row index where actual data begins (time + sensor values)
CORE_LIST = [f"Core {i}" for i in range(26)]  # Expected core column headers


# -----------------------------
# Data extraction
# -----------------------------
def extract_core_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts per-core temperature data from a raw OCCT DataFrame.

    - Finds columns whose header row (row index 1) matches CORE_LIST.
    - Converts the first column to numeric time (seconds). If timestamps look like
      epoch ms (>1e6), convert to seconds.
    - Normalizes time to start at 0 s and uses it as DataFrame index.
    - Returns a DataFrame of core temps (float), indexed by time (seconds).
    """
    # Identify core columns by header names in row 1
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")

    # Parse time column (first column). Handle epoch ms gracefully.
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
    if time.iloc[0] > 1e6:  # crude heuristic for ms timestamps
        time = time / 1000.0
    time = time - time.iloc[0]  # normalize to start at zero

    # Parse core temperatures; coerce errors to NaN (keeps array math safe)
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]

    return core_data


# -----------------------------
# Correlation heatmap
# -----------------------------
def generate_core_correlation_plot(core_df: pd.DataFrame, title: str):
    """
    Computes the correlation matrix between cores and returns a Matplotlib heatmap
    figure along with the raw correlation DataFrame.
    """
    corr = core_df.corr()  # Pearson correlation between core columns

    # Build annotated heatmap with compact formatting (one sig. figure)
    fig, ax = plt.subplots(figsize=(10, 8))
    formatted_corr = corr.applymap(lambda x: f"{x:.1g}" if pd.notnull(x) else "")
    sns.heatmap(
        corr,
        annot=formatted_corr, fmt="", cmap='coolwarm',
        vmin=-1, vmax=1, square=True,
        annot_kws={"size": 12},
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, corr


# -----------------------------
# Graph construction from correlations
# -----------------------------
def build_dependency_graph(corr_matrix: pd.DataFrame, threshold: float = 0.9) -> nx.Graph:
    """
    Creates an undirected graph where nodes are cores and edges connect cores
    whose correlation is >= threshold.
    """
    G = nx.Graph()

    # Add nodes
    for core in corr_matrix.columns:
        G.add_node(core)

    # Add edges for strong correlations
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            c1, c2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if pd.notnull(corr_val) and corr_val >= threshold:
                G.add_edge(c1, c2, weight=round(float(corr_val), 2))

    return G


# -----------------------------
# Plotly graph visualization
# -----------------------------
def plot_dependency_graph_plotly(G: nx.Graph, title: str, threshold: float = 0.9) -> go.Figure:
    """
    Plots the dependency graph using Plotly:
    - Spring layout for node positions (stable via fixed seed).
    - Edges with hover tooltips showing correlation.
    - Nodes labeled by core and hover showing neighbors + edge weights.
    """
    # Compute positions via spring layout (deterministic thanks to seed)
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=100)

    # ----- Edge traces -----
    edge_x, edge_y, edge_weights, edge_texts = [], [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(data['weight'])
        edge_texts.append(f"{u} ↔ {v}<br>Corr: {data['weight']:.2f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        hoverinfo='text', text=edge_texts,
        line=dict(width=2, color='rgba(150, 0, 0, 0.6)')
    )

    # ----- Node traces -----
    node_x, node_y, node_hover, node_labels = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        connections = [f"{nbr} (Corr: {G[node][nbr]['weight']:.2f})" for nbr in G.adj[node]]
        hover_text = f"{node}<br>Connected to: " + (", ".join(connections) if connections else "None")
        node_hover.append(hover_text)
        node_labels.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        textposition='top center',
        marker=dict(size=20, line=dict(width=2, color='DarkSlateGrey')),
        hovertext=node_hover
    )

    # Compose final figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=f"🔗 {title} (thr={threshold})", font=dict(size=16)),
            showlegend=False, hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40), height=700,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    )
    return fig


# -----------------------------
# Streamlit app entry point
# -----------------------------
def run_core_correlation_matrix():
    """
    Streamlit workflow:
    - Upload multiple CSVs (each processed independently).
    - For each file: extract cores → correlation heatmap (+PNG download)
      → dependency graph with threshold slider (+HTML download).
    """
    st.subheader("🧩 Multi-Test Core Correlation & Dependency Graphs")

    # Allow multiple files; each gets its own analysis block
    uploaded_files = st.file_uploader(
        "📂 Upload OCCT CSV Files (multiple)", type=["csv"], accept_multiple_files=True, key="corr-upload"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown(f"---\n### Results for **{uploaded_file.name}**")
            try:
                # Load raw file
                df = pd.read_csv(uploaded_file, header=None)

                # Extract per-core temps with normalized time index
                core_df = extract_core_data(df)

                # ----- Correlation heatmap (Matplotlib) -----
                corr_fig, corr_matrix = generate_core_correlation_plot(
                    core_df, f"Core Correlation: {uploaded_file.name}"
                )
                st.pyplot(corr_fig)

                # Download correlation heatmap PNG
                corr_buf = BytesIO()
                corr_fig.savefig(corr_buf, format="png")
                st.download_button(
                    label="💾 Download Heatmap PNG",
                    data=corr_buf.getvalue(),
                    file_name=f"{uploaded_file.name.replace('.', '_')}_heatmap.png",
                    mime="image/png",
                    key=f"dl-heatmap-{uploaded_file.name}"
                )

                # ----- Dependency graph threshold -----
                # Exclude diagonal when computing min correlation for slider bounds
                corr_values = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool)).stack()
                min_corr = float(corr_values.min()) if not corr_values.empty else 0.0

                thr_key = f"thr-{uploaded_file.name}"
                threshold = st.slider(
                    f"Threshold for {uploaded_file.name}",
                    min_value=round(min_corr, 2),  # dynamic lower bound based on data
                    max_value=1.0,
                    value=0.9,
                    step=0.01,
                    key=thr_key
                )

                # Build graph and warn if nothing passes threshold
                G = build_dependency_graph(corr_matrix, threshold=threshold)
                if G.number_of_edges() == 0:
                    st.warning("⚠️ No edges to display at this threshold.")

                # ----- Plotly interactive graph -----
                graph_fig = plot_dependency_graph_plotly(
                    G, f"Dependency Graph: {uploaded_file.name}", threshold=threshold
                )
                st.plotly_chart(graph_fig, use_container_width=True)

                # Download Plotly graph as standalone HTML
                html_bytes = pio.to_html(graph_fig, full_html=True).encode('utf-8')
                st.download_button(
                    label="📥 Download Graph HTML",
                    data=html_bytes,
                    file_name=f"{uploaded_file.name}_dependency_graph.html",
                    mime="text/html",
                    key=f"dl-graph-{uploaded_file.name}"
                )

            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")