import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
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
                vmin=-1, vmax=1, square=True,
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
    ax.set_title(f"Core Correlation Matrix\n{filename}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def run_core_correlation_matrix():
    st.subheader("🧩 Core Correlation Matrix")

    uploaded_file = st.file_uploader("📂 Upload OCCT CSV File", type=["csv"], key="corr-upload")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            core_df = extract_core_data(df)
            fig = generate_core_correlation_plot(core_df, uploaded_file.name)

            st.pyplot(fig)

            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.download_button(
                label="💾 Download Correlation Matrix as PNG",
                data=buf.getvalue(),
                file_name=f"{uploaded_file.name.replace('.', '_')}_correlation_matrix.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")