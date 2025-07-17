import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.manifold import MDS
from io import BytesIO

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(25)]  # Intel W7-2595X has 26 cores, but only 25 show up in OCCT

def extract_core_data(df):
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
    time = time - time.iloc[0]
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]
    return core_data

def run_core_physical_layout():
    st.subheader("📐 Estimated Core Physical Layout (MDS based)")

    uploaded_files = st.file_uploader(
        "📂 Upload one or more OCCT CSV Files",
        type=["csv"],
        key="layout-upload",
        accept_multiple_files=True
    )
    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        st.markdown(f"### 📄 {uploaded_file.name}")
        try:
            df = pd.read_csv(uploaded_file, header=None)
            core_df = extract_core_data(df)

            # Compute correlation matrix
            corr_matrix = core_df.corr()

            # Convert correlation to distance
            dist_matrix = 1 - corr_matrix

            # Apply MDS
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            positions = mds.fit_transform(dist_matrix)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 8))
            for i, (x, y) in enumerate(positions):
                ax.scatter(x, y, s=100, color='skyblue', edgecolors='black')
                ax.text(x, y, f"Core {i}", fontsize=9, ha='center', va='center')
            ax.set_title("🧭 Estimated Physical Layout of CPU Cores (based on temperature correlation)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)

            st.pyplot(fig)

            # Optional download
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                f"📥 Download Layout Plot ({uploaded_file.name})",
                buf.getvalue(),
                file_name=f"{uploaded_file.name}_core_physical_layout.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
