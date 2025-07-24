import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Config ----------
START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]

# ---------- Helpers ----------
def extract_core_data(df):
    # Identify core temperature columns
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")
    # Time index
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
    time = time - time.iloc[0]
    # Extract and clean core data
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]
    return core_data

# ---------- App ----------
def main():
    st.title("📊 PCA & Clustering of CPU Test Runs")
    st.markdown("Upload multiple OCCT/CSV/XLSX files to compare them using PCA and K-Means clustering.")

    files = st.file_uploader(
        "📂 Upload OCCT Files (select multiple)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        key="pca_clustering_upload"
    )
    if not files or len(files) < 2:
        st.info("Please upload at least two files to compare.")
        return

    # Process each file to get average core temperatures
    avg_temps = {}
    for f in files:
        try:
            # Read file
            if f.name.endswith('.csv'):
                df = pd.read_csv(f, header=None)
            else:
                df = pd.read_excel(f, header=None, engine='openpyxl')
            core_df = extract_core_data(df)
            # Compute per-core average
            avg = core_df.mean(axis=0)
            avg_temps[f.name] = avg.values
        except Exception as e:
            st.error(f"Error processing `{f.name}`: {e}")
            return

    # Build DataFrame: rows = file names, columns = cores
    data = pd.DataFrame(avg_temps).T
    data.columns = CORE_LIST
    st.subheader("Average Core Temperatures per Test")
    st.dataframe(data)
    # Download average core temps
    csv_avg = data.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Average Core Temperatures (CSV)",
        data=csv_avg,
        file_name="avg_core_temperatures.csv",
        mime="text/csv"
    )

    # Standardize the data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # PCA projection
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)
    pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=data.index)

    # K-Means clustering
    k = st.slider(
        "🔢 Number of Clusters (K)",
        min_value=2,
        max_value=min(len(files), 10),
        value=3
    )
    km = KMeans(n_clusters=k, random_state=0)
    clusters = km.fit_predict(scaled)
    pc_df["Cluster"] = clusters.astype(str)

    # Plot PCA scatter colored by cluster
    st.subheader("PCA Projection with K-Means Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=pc_df['PC1'],
        y=pc_df['PC2'],
        hue=pc_df['Cluster'],
        palette='tab10',
        s=100,
        ax=ax
    )
    for idx, row in pc_df.iterrows():
        ax.text(row.PC1 + 0.02, row.PC2 + 0.02, idx, fontsize=9)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA of Test Runs (colored by cluster)")
    st.pyplot(fig)

    # Display clusters table and download
    st.subheader("Cluster Assignments")
    st.table(pc_df[['Cluster']])
    csv_pca = pc_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download PCA & Cluster Data (CSV)",
        data=csv_pca,
        file_name="pca_cluster_data.csv",
        mime="text/csv"
    )

    # Show explained variance and download
    st.subheader("PCA Explained Variance Ratio")
    ev_df = pd.DataFrame(
        pca.explained_variance_ratio_.reshape(1, -1),
        columns=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))]
    )
    st.write(ev_df)
    csv_ev = ev_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Explained Variance (CSV)",
        data=csv_ev,
        file_name="explained_variance.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()