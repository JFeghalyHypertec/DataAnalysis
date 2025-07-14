import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
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

def plot_heatmap(core_df, file_path):
    averages = core_df[core_df != 0].mean()
    overall_avg = averages.mean()

    fig = plt.figure(figsize=(16, 8))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[4, 1])

    ax0 = fig.add_subplot(spec[0])
    sns.heatmap(core_df.T, cmap="coolwarm", cbar_kws={'label': 'Temperature (°C)'}, ax=ax0)
    file_name = file_path.name
    ax0.set_title(f"Core Temperatures Over Time (Heatmap)\n {file_name}")
    core_df.index /= 3600  # Convert index to hours
    ax0.set_xlabel("Time (hours)")
    ax0.set_ylabel("CPU Cores")

    ax1 = fig.add_subplot(spec[1])
    bars = ax1.barh(averages.index, averages.values, color='gray')
    ax1.set_title("Avg Temp per Core")
    ax1.set_xlim(averages.min() - 5, averages.max() + 5)
    ax1.set_xlabel("°C")

    for bar, value in zip(bars, averages.values):
        ax1.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}°C",
                 va='center', ha='left', fontsize=9)

    ax1.text(
        0.5, 1.05,
        f"Overall Avg Temp: {overall_avg:.1f}°C",
        ha='center', va='center',
        transform=ax1.transAxes,
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightyellow')
    )
    plt.tight_layout()
    return fig

def run_core_heatmap_plot():
    st.header("🔍 Core Heatmap Plot")

    uploaded_files = st.file_uploader(
        "Upload one or more CSV or Excel files",
        type=['csv', 'xls', 'xlsx'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, header=None)
                else:
                    df = pd.read_excel(uploaded_file, header=None)

                core_df = extract_core_data(df)
                fig = plot_heatmap(core_df, uploaded_file)

                st.subheader(f"📊 Heatmap for: `{file_name}`")
                st.pyplot(fig)

                # Save button with dynamic filename
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label=f"💾 Download Heatmap for {file_name}",
                    data=buf.getvalue(),
                    file_name=f"{file_name.replace('.','_')}_heatmap.png",
                    mime="image/png"
                )
                st.markdown("---")  # Divider between plots

            except Exception as e:
                st.error(f"❌ Error processing {file_name}: {str(e)}")