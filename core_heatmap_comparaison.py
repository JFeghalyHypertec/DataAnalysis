import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import matplotlib.gridspec as gridspec
from pathlib import Path

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]
TR1 = "TR1 Temperature (System Board)"
CPU_PACKAGE = "CPU Package"
WATER_FLOW = "Water Flow"

def extract_core_data(df):
    time_col = 0
    time = pd.to_numeric(df.iloc[START_ROW:, time_col], errors='coerce')

    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]
    core_data = core_data.replace(0, np.nan)

    tr1_col = next((i for i in range(df.shape[1]) if df.iloc[1, i] == TR1), None)
    if tr1_col is not None:
        tr1_data = pd.to_numeric(df.iloc[START_ROW:, tr1_col], errors='coerce')
        tr1_data.index = time
        tr1_data = tr1_data.replace(0, np.nan)
        tr1_grouped = tr1_data.groupby((tr1_data.index // 60) * 60).mean()
    else:
        tr1_grouped = None

    core_data['time_bucket'] = (core_data.index // 60) * 60
    grouped_core = core_data.groupby('time_bucket').mean().dropna(axis=1, how='all')
    return grouped_core, tr1_grouped

def get_col(df, name):
    try:
        return next(i for i in range(df.shape[1]) if df.iloc[1, i] == name)
    except StopIteration:
        return None

def get_numeric_col(df, name):
    col_idx = get_col(df, name)
    if col_idx is None:
        return pd.Series(dtype=float)
    vals = pd.to_numeric(df.iloc[START_ROW:, col_idx], errors='coerce')
    return vals[(vals != 0) & (~vals.isna())]

def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, header=None)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file, header=None, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}")
    except Exception as e:
        st.error(f"❌ Failed to read file `{uploaded_file.name}`: {e}")
        return None

def run_core_heatmap_comparaison():
    st.header("🔥 Core Difference Heatmap")
    file1 = st.file_uploader("Upload the FIRST CPU data file", type=["csv","xls","xlsx"], key="file1_cmp")
    file2 = st.file_uploader("Upload the SECOND CPU data file", type=["csv","xls","xlsx"], key="file2_cmp")
    st.info("ℹ️ Recommended: hottest test as second, coldest as first for positive ΔTemp.")
    if not file1 or not file2:
        return

    raw1 = read_uploaded_file(file1)
    raw2 = read_uploaded_file(file2)
    if raw1 is None or raw2 is None:
        return

    file1_name, file2_name = file1.name, file2.name

    core1, tr1_1 = extract_core_data(raw1)
    core2, tr1_2 = extract_core_data(raw2)

    plate1 = st.text_input(f"Plate for {file1_name}:", key="plate1_cmp").strip() or "NA"
    occt1  = st.text_input(f"OCCT Version for {file1_name}:", key="occt1_cmp").strip() or "NA"
    plate2 = st.text_input(f"Plate for {file2_name}:", key="plate2_cmp").strip() or "NA"
    occt2  = st.text_input(f"OCCT Version for {file2_name}:", key="occt2_cmp").strip() or "NA"

    cpu1 = get_numeric_col(raw1, CPU_PACKAGE)
    cpu2 = get_numeric_col(raw2, CPU_PACKAGE)
    cpu1_avg = cpu1.mean() if not cpu1.empty else np.nan
    cpu2_avg = cpu2.mean() if not cpu2.empty else np.nan
    cpu_diff = cpu2_avg - cpu1_avg if pd.notnull(cpu1_avg) and pd.notnull(cpu2_avg) else np.nan

    overall1 = np.nanmean(core1.values.flatten()) if core1.size else np.nan
    overall2 = np.nanmean(core2.values.flatten()) if core2.size else np.nan
    overall_diff = overall2 - overall1 if pd.notnull(overall1) and pd.notnull(overall2) else np.nan

    wf1 = get_numeric_col(raw1, WATER_FLOW)
    wf2 = get_numeric_col(raw2, WATER_FLOW)
    wf1_avg = wf1.mean() if not wf1.empty else np.nan
    wf2_avg = wf2.mean() if not wf2.empty else np.nan
    wf_diff = wf2_avg - wf1_avg if pd.notnull(wf1_avg) and pd.notnull(wf2_avg) else np.nan

    cols = ["", "Plate", "OCCT Version", f"{CPU_PACKAGE} Temperature", "Overall Avg Core Temp", WATER_FLOW]
    data = [
        [file1_name, plate1, occt1, round(cpu1_avg,2) if pd.notnull(cpu1_avg) else "NA", round(overall1,2) if pd.notnull(overall1) else "NA", round(wf1_avg,2) if pd.notnull(wf1_avg) else "NA"],
        [file2_name, plate2, occt2, round(cpu2_avg,2) if pd.notnull(cpu2_avg) else "NA", round(overall2,2) if pd.notnull(overall2) else "NA", round(wf2_avg,2) if pd.notnull(wf2_avg) else "NA"],
        ["File2 - File1", "", "", round(cpu_diff,2) if pd.notnull(cpu_diff) else "NA", round(overall_diff,2) if pd.notnull(overall_diff) else "NA", round(wf_diff,2) if pd.notnull(wf_diff) else "NA"]
    ]
    summary_df = pd.DataFrame(data, columns=cols)
    st.subheader("📋 Comparison Summary")

    t1, t2 = core1.index.max(), core2.index.max()
    if abs(t1 - t2) > 60:
        st.error(f"❗ Time misalignment: {abs(t1-t2)}s")
        return
    idx = core1.index.intersection(core2.index)
    cols_common = core1.columns.intersection(core2.columns)
    a1 = core1.loc[idx, cols_common]
    a2 = core2.loc[idx, cols_common]
    if tr1_1 is not None and tr1_2 is not None:
        tr1_1, tr1_2 = tr1_1.loc[idx], tr1_2.loc[idx]
        a1 = a1.subtract(tr1_1, axis=0)
        a2 = a2.subtract(tr1_2, axis=0)

    df_diff = a2 - a1
    df_diff.index = (df_diff.index/3600).round(2)

    fig = plt.figure(figsize=(16,12))
    spec = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[4,1])
    ax0 = fig.add_subplot(spec[0,0])
    sns.heatmap(df_diff.T, cmap="coolwarm", center=0, cbar_kws={'label':'ΔTemp (°C)'}, ax=ax0)
    ax0.set_title(f"Diff Heatmap: {file2_name} - {file1_name}")
    ax0.set_xlabel("Time (h)")
    ax0.set_ylabel("CPU Cores")

    avgs = df_diff[df_diff!=0].mean().round(2)
    overall = avgs.mean().round(2)
    ax1 = fig.add_subplot(spec[0,1])
    colors = ['red' if x>0 else 'blue' if x<0 else 'gray' for x in avgs.values]
    bars = ax1.barh(avgs.index, avgs.values, color=colors)
    ax1.set_title("Avg ΔTemp per Core")
    ax1.set_xlim(avgs.min()-5, avgs.max()+5)
    ax1.set_xlabel("°C")
    for b,v in zip(bars, avgs.values):
        ax1.text(v+0.5, b.get_y()+b.get_height()/2, f"{v}°C", va='center')
    ax1.text(0.5,1.05,f"Overall Avg Δ: {overall}°C", ha='center', va='center', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3",facecolor='lightyellow',edgecolor='black'))

    ax_table = fig.add_subplot(spec[1,:])
    ax_table.axis("off")
    table = ax_table.table(cellText=summary_df.values, colLabels=summary_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    st.pyplot(fig)
    st.table(summary_df)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    st.download_button("📥 Download Difference Heatmap", data=buf.getvalue(), file_name="difference_heatmap.png", mime="image/png")