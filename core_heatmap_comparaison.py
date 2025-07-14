import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import os

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]

def extract_core_data(df):
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    time_col = 0
    time = pd.to_numeric(df.iloc[START_ROW:, time_col], errors='coerce')
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]

    core_data = core_data.replace(0, np.nan)
    core_data['time_bucket'] = (core_data.index // 60) * 60
    grouped = core_data.groupby('time_bucket').mean()

    return grouped.dropna(axis=1, how='all')

def run_core_heatmap_comparaison():
    file1 = st.file_uploader("Upload the FIRST CPU data file", type=["csv", "xls", "xlsx"], key="file1")
    file2 = st.file_uploader("Upload the SECOND CPU data file", type=["csv", "xls", "xlsx"], key="file2")
    try:
        file1_name = os.path.basename(file1.name)
        file2_name = os.path.basename(file2.name)

        df1 = pd.read_csv(file1, header=None) if file1.name.endswith(".csv") else pd.read_excel(file1, header=None)
        df2 = pd.read_csv(file2, header=None) if file2.name.endswith(".csv") else pd.read_excel(file2, header=None)

        df1 = extract_core_data(df1)
        df2 = extract_core_data(df2)

        # Validate time alignment
        t1 = df1.index.max()
        t2 = df2.index.max()
        if abs(t1 - t2) > 60:
            st.warning(f"❗ Time misalignment detected.\nLast timestamp difference: {abs(t1 - t2)} seconds")

        common_index = df1.index.intersection(df2.index)
        common_columns = df1.columns.intersection(df2.columns)

        df1_aligned = df1.loc[common_index, common_columns]
        df2_aligned = df2.loc[common_index, common_columns]
        df_diff = df2_aligned - df1_aligned

        st.subheader("🧊 Temperature Difference Heatmap")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(df_diff.T, cmap="RdBu", center=0, cbar_kws={'label': 'ΔTemp (°C)'}, ax=ax)

        title = f"Difference Heatmap ({file2_name} - {file1_name})\nAveraged every 60 seconds"
        ax.set_title(title)
        ax.set_xlabel("Time Bucket (s)")
        ax.set_ylabel("CPU Cores")
        st.pyplot(fig)

        # Download option
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("📥 Download Difference Heatmap", data=buf.getvalue(),
                           file_name="difference_heatmap.png", mime="image/png")

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")