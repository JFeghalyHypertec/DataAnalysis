import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
from io import BytesIO
from sklearn.decomposition import PCA

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
    st.subheader("🧠 Core Temperature Clustering")

    uploaded_file = st.file_uploader("📂 Upload OCCT CSV File", type=["csv"], key="clustering-upload")
    if not uploaded_file:
        return

    try:
        df = pd.read_csv(uploaded_file, header=None)
        core_df = extract_core_data(df)

        # Transpose: rows = cores, columns = time snapshots
        core_profiles = core_df.T
        core_profiles_clean = core_profiles.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
        core_profiles_scaled = StandardScaler().fit_transform(core_profiles_clean)

        # Clustering
        k = st.slider("🔢 Number of Clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(core_profiles_scaled)

        # Results
        cluster_df = pd.DataFrame({
            "Core": core_profiles.index,
            "Cluster": labels
        })

        st.dataframe(cluster_df.sort_values("Cluster"))

        # Plot clusters (2D projection)
        pca = PCA(n_components=2)
        points = pca.fit_transform(core_profiles_scaled)

        fig, ax = plt.subplots()
        for cluster_id in range(k):
            idx = labels == cluster_id
            ax.scatter(points[idx, 0], points[idx, 1], label=f"Cluster {cluster_id}")
        for i, name in enumerate(core_profiles.index):
            ax.annotate(name, (points[i, 0], points[i, 1]), fontsize=8, alpha=0.6)
        ax.set_title(f"🌀 Core Clusters (K={k})")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")