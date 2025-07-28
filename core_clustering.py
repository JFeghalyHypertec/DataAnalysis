import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
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

def run_core_clustering():
    st.subheader("📊 PCA + Core Clustering")

    uploaded_file = st.file_uploader("📂 Upload OCCT CSV File", type=["csv"], key="pca-cluster-upload")
    if not uploaded_file:
        return

    try:
        df = pd.read_csv(uploaded_file, header=None)
        core_df = extract_core_data(df).dropna(axis=1, how='any')

        if core_df.shape[1] < 2:
            st.warning("📉 Not enough valid core columns after cleaning.")
            return

        n_clusters = st.slider("🔢 Number of Clusters", 2, min(10, len(core_df.columns)), 4)

        # PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(core_df.T)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(reduced)
        core_names = core_df.columns

        # Scatter plot
        fig1, ax1 = plt.subplots()
        scatter = ax1.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=100)
        for i, txt in enumerate(core_names):
            ax1.annotate(txt, (reduced[i, 0], reduced[i, 1]), fontsize=9, ha='right')
        ax1.set_title("PCA Projection with Core Clusters")
        ax1.set_xlabel("Principal Component 1")
        ax1.set_ylabel("Principal Component 2")
        ax1.grid(True)
        st.pyplot(fig1)

        # Download cluster data
        cluster_df = pd.DataFrame({
            "Core": core_names,
            "Cluster": labels
        }).sort_values(by="Cluster")
        st.dataframe(cluster_df, use_container_width=True)

        csv = cluster_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cluster Assignments (CSV)", data=csv,
                           file_name="core_clusters.csv", mime="text/csv")

        # Clustered cosine distance heatmap
        sorted_idx = np.argsort(labels)
        sorted_cores = core_names[sorted_idx]
        sorted_reduced = reduced[sorted_idx]
        dist_matrix = squareform(pdist(sorted_reduced, metric="cosine"))
        dist_df = pd.DataFrame(dist_matrix, index=sorted_cores, columns=sorted_cores)

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(dist_df, cmap="mako", square=True, ax=ax2, cbar_kws={'label': 'Cosine Distance'})
        ax2.set_title("Clustered Core Distance Heatmap (PCA Space)")
        st.pyplot(fig2)

        # Download heatmap
        buf = BytesIO()
        fig2.savefig(buf, format="png")
        st.download_button("📥 Download Clustered Heatmap", data=buf.getvalue(),
                           file_name="clustered_core_heatmap.png", mime="image/png")

    except Exception as e:
        st.error(f"❌ Error: {e}")