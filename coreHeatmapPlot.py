import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
from io import BytesIO
from pathlib import Path

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]
CPU_PACKAGE = "CPU Package"
WATER_FLOW = "Water Flow"

def extract_core_data(df):
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time

    core_data.columns = [df.iloc[1, i] for i in core_cols]
    return core_data

def plot_heatmap(core_df, file_path, summary_table=None):
    averages = core_df[core_df != 0].mean()
    overall_avg = averages.mean()

    # Increased height for the embedded table
    fig = plt.figure(figsize=(16, 10))
    spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[4, 1], height_ratios=[4, 1])

    # Heatmap
    ax0 = fig.add_subplot(spec[0, 0])
    core_df.index = (core_df.index / 3600).round(2)  # Convert index to hours
    sns.heatmap(core_df.T, cmap="coolwarm", cbar_kws={'label': 'Temperature (°C)'}, ax=ax0)
    file_name = file_path.name
    ax0.set_title(f"Core Temperatures Over Time (Heatmap)\n {file_name}")
    ax0.set_xlabel("Time (hours)")
    ax0.set_ylabel("CPU Cores")

    # Bar Chart
    ax1 = fig.add_subplot(spec[0, 1])
    bars = ax1.barh(averages.index, averages.values, color='gray')
    ax1.set_title("Avg Temp per Core")
    ax1.set_xlim(averages.min() - 5, averages.max() + 5)
    ax1.set_xlabel("°C")

    for bar, value in zip(bars, averages.values):
        ax1.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}°C",
                 va='center', ha='left', fontsize=9)
        
    # Summary Table
    ax2 = fig.add_subplot(spec[1, :])
    ax2.axis("off")
    if summary_table is not None:
        cell_text = summary_table.values
        col_labels = summary_table.columns
        table = ax2.table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

    plt.tight_layout()
    return fig

def get_col(df, name):
    try:
        return next(i for i in range(df.shape[1]) if df.iloc[1, i] == name)
    except StopIteration:
        return None

def get_numeric_col(df, name):
    col_idx = get_col(df, name)
    if col_idx is None: return None
    values = pd.to_numeric(df.iloc[START_ROW:, col_idx], errors='coerce')
    return values[(values != 0) & (~values.isna())]

def build_summary_table(df, core_df, file_path):
    """Return a 1-row DataFrame with Water Flow, CPU Package, Avg Core, Plate, OCCT ver."""

    water_flow   = round(get_numeric_col(df, WATER_FLOW).mean(), 1) if get_col(df, WATER_FLOW) is not None else None
    cpu_package  = round(get_numeric_col(df, CPU_PACKAGE).mean(), 1) if get_col(df, CPU_PACKAGE) is not None else None
    avg_core_tmp = core_df[core_df != 0].mean().mean()

    parent_name = Path(file_path.name).parent.name
    parts = parent_name.split("-") if parent_name else []
    plate = parts[0] if parts else None
    occt_version = next((p.replace("OCCT", "") for p in parts if p.startswith("OCCT")), None)

    # ── convert to strings, “NA” if missing ────────────────────────────────────
    wf_str  = f"{water_flow:.2f} L/h" if not np.isnan(water_flow) else "NA"     # --
    cpu_str = f"{cpu_package:.2f} °C" if not np.isnan(cpu_package) else "NA"   # --
    avg_str = f"{avg_core_tmp:.2f} °C"                                          # always exists
    plate   = plate or "NA"                                                  # --
    occt_version = occt_version or "NA"                                      # --

    columns = ["Water Flow", "CPU Package", "Overall Avg Core", "Plate", "OCCT Version"]
    values  = [wf_str, cpu_str, avg_str, plate, occt_version]

    return pd.DataFrame([values], columns=columns)

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
                
                st.subheader(f"📊 Heatmap for: `{file_name}`")
                summary = build_summary_table(df, core_df, uploaded_file)
                fig = plot_heatmap(core_df, uploaded_file, summary_table=summary)
                st.pyplot(fig)
                st.table(summary)
                
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