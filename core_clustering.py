import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

# Elbow and Silhouette utilities
def compute_elbow_curve(data, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(data)
        inertias.append(km.inertia_)
    return inertias

def compute_silhouette_scores(data, max_k=10):
    scores = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=0)
        labels = km.fit_predict(data)
        scores.append(silhouette_score(data, labels))
    return scores

# Main function

def run_core_clustering():
    st.subheader("📊 PCA + Core Clustering")

    uploaded_file = st.file_uploader("📂 Upload OCCT CSV File", type=["csv"], key="pca-cluster-upload")
    if not uploaded_file:
        return

    try:
        # Read and clean data
        df = pd.read_csv(uploaded_file, header=None)
        core_df = extract_core_data(df).dropna(axis=1, how='any')
        if core_df.shape[1] < 2:
            st.warning("📉 Not enough valid core columns after cleaning.")
            return

        # PCA reduction
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(core_df.T)

        # Suggest optimal k via silhouette
        st.subheader("📌 Suggested Optimal Cluster Count")
        max_k = st.slider("🔍 Max clusters to test", 3, min(15, len(core_df.columns)), min(8, len(core_df.columns)), key="max-k")
        sil_scores = compute_silhouette_scores(reduced, max_k=max_k)
        ks = list(range(2, max_k + 1))
        best_k = ks[np.argmax(sil_scores)]
        best_score = max(sil_scores)
        st.markdown(f"🎯 **Optimal k:** {best_k} (Silhouette = {best_score:.3f})")

        # Plot silhouette vs k
        fig_sil, ax_sil = plt.subplots()
        ax_sil.plot(ks, sil_scores, marker='o')
        ax_sil.set_title("Silhouette Score vs. Number of Clusters")
        ax_sil.set_xlabel("k")
        ax_sil.set_ylabel("Silhouette Score")
        st.pyplot(fig_sil)

        # Select clusters (default to best_k)
        n_clusters = st.slider("🔢 Number of Clusters", 2, min(10, len(core_df.columns)), value=best_k, key="n-clusters")

        # KMeans clustering and plotting
        km = KMeans(n_clusters=n_clusters, random_state=0)
        labels = km.fit_predict(reduced)
        cores = list(core_df.columns)

        fig1, ax1 = plt.subplots()
        scatter = ax1.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=100)
        for i, core in enumerate(cores):
            ax1.annotate(core, (reduced[i, 0], reduced[i, 1]), fontsize=9, ha='right')
        ax1.set_title("PCA Projection with Core Clusters")
        ax1.set_xlabel("PC 1")
        ax1.set_ylabel("PC 2")
        ax1.grid(True)
        st.pyplot(fig1)

        # Show and download cluster assignments
        cluster_df = pd.DataFrame({"Core": cores, "Cluster": labels}).sort_values("Cluster")
        st.dataframe(cluster_df)
        st.download_button("📥 Download Clusters CSV", data=cluster_df.to_csv(index=False).encode(),
                           file_name="core_clusters.csv", mime="text/csv")

        # Clustered distance heatmap
        idx = np.argsort(labels)
        sorted_cores = [cores[i] for i in idx]
        sorted_reduced = reduced[idx]
        dist = squareform(pdist(sorted_reduced, metric="cosine"))
        dist_df = pd.DataFrame(dist, index=sorted_cores, columns=sorted_cores)
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(dist_df, cmap="mako", square=True, cbar_kws={'label':'Cosine Distance'}, ax=ax2)
        ax2.set_title("Clustered Core Distance Heatmap (PCA)")
        st.pyplot(fig2)

        # Download heatmap
        buf = BytesIO()
        fig2.savefig(buf, format="png")
        st.download_button("📥 Download Heatmap PNG", data=buf.getvalue(),
                           file_name="clustered_heatmap.png", mime="image/png")

    except Exception as e:
        st.error(f"❌ Error: {e}")
