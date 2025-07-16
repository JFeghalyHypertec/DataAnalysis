import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import matplotlib.gridspec as gridspec

START_ROW = 3
CORE_LIST = [f"Core {i}" for i in range(26)]
TR1 = "TR1 Temperature (System Board)"

def extract_core_data(df):
    time_col = 0
    time = pd.to_numeric(df.iloc[START_ROW:, time_col], errors='coerce')

    # Extract core temperature columns
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]
    core_data = core_data.replace(0, np.nan)

    # Extract TR1 column
    tr1_col = next((i for i in range(df.shape[1]) if df.iloc[1, i] == TR1), None)
    if tr1_col is not None:
        tr1_data = pd.to_numeric(df.iloc[START_ROW:, tr1_col], errors='coerce')
        tr1_data.index = time
        tr1_data = tr1_data.replace(0, np.nan)
        tr1_grouped = tr1_data.groupby((tr1_data.index // 60) * 60).mean()
    else:
        tr1_grouped = None

    # Group core data
    core_data['time_bucket'] = (core_data.index // 60) * 60
    grouped_core = core_data.groupby('time_bucket').mean().dropna(axis=1, how='all')

    return grouped_core, tr1_grouped


def run_core_heatmap_comparaison():
    file1 = st.file_uploader("Upload the FIRST CPU data file", type=["csv", "xls", "xlsx"], key="file1")
    file2 = st.file_uploader("Upload the SECOND CPU data file", type=["csv", "xls", "xlsx"], key="file2")
    st.info("ℹ️ **Recommended:** Upload the **hottest test** as the **second file** and the **coldest test** as the **first file**.\nThis ensures that temperature differences are positive when cooling is effective.")

    if not file1 or not file2:
        return
    
    if file1.name == file2.name:
        #st.error("❗ Please upload two different files for comparison.")
        #return
        y = 1
        
    try:
        file1_name = file1.name
        file2_name = file2.name

        df1 = pd.read_csv(file1, header=None) if file1.name.endswith(".csv") else pd.read_excel(file1, header=None)
        df2 = pd.read_csv(file2, header=None) if file2.name.endswith(".csv") else pd.read_excel(file2, header=None)

        df1, tr1_df1 = extract_core_data(df1)
        df2, tr1_df2 = extract_core_data(df2)

        # Validate time alignment
        t1 = df1.index.max()
        t2 = df2.index.max()
        if abs(t1 - t2) > 60:
            st.error(f"❗ Time misalignment detected.\nLast timestamp difference: {abs(t1 - t2)} seconds")
            return
        common_index = df1.index.intersection(df2.index)
        common_columns = df1.columns.intersection(df2.columns)

        df1_aligned = df1.loc[common_index, common_columns]
        df2_aligned = df2.loc[common_index, common_columns]


        if tr1_df1 is not None and tr1_df2 is not None:
            tr1_df1 = tr1_df1.loc[common_index]
            tr1_df2 = tr1_df2.loc[common_index]

            df1_aligned = df1_aligned.subtract(tr1_df1, axis=0)
            df2_aligned = df2_aligned.subtract(tr1_df2, axis=0)
        else:
            missing_in = []
            if tr1_df1 is None:
                missing_in.append(f"❌ TR1 not found in **{file1_name}**")
            if tr1_df2 is None:
                missing_in.append(f"❌ TR1 not found in **{file2_name}**")
            st.warning("⚠️ TR1 adjustment skipped:\n" + "\n".join(missing_in))

        
        df_diff = df2_aligned - df1_aligned
        df_diff.index = (df_diff.index / 3600).round(2)

        st.subheader("🧊 Temperature Difference Heatmap")
        fig = plt.figure(figsize=(16, 8))
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[4, 1])
        
        ax0 = fig.add_subplot(spec[0])
        sns.heatmap(df_diff.T, cmap="coolwarm", center=0, cbar_kws={'label': 'ΔTemp (°C)'}, ax=ax0)

        title = f"Difference Heatmap ({file2_name} - {file1_name})\nAveraged every 60 seconds"
        ax0.set_title(title)
        ax0.set_xlabel("Time (hours)")
        ax0.set_ylabel("CPU Cores")
        
        averages = df_diff[df_diff != 0].mean().round(2)
        overall_avg = averages.mean().round(2)
        ax1 = fig.add_subplot(spec[1])
        colors = ['red' if x > 0 else 'blue' if x < 0 else 'gray' for x in averages.values]
        bars = ax1.barh(averages.index, averages.values, color=colors)
        ax1.set_title("Avg Temp Difference per Core")
        ax1.set_xlim(averages.min() - 5, averages.max() + 5)
        ax1.set_xlabel("°C")
        for bar, value in zip(bars, averages.values):
            ax1.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}°C",
                     va='center', ha='left', fontsize=9)
            
        ax1.text(
            0.5, 1.05,
            f"Overall Avg Temp Difference: {overall_avg:.1f}°C",
            ha='center', va='center',
            transform=ax1.transAxes,
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightyellow')
        )
        st.pyplot(fig)

        # Download option
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("📥 Download Difference Heatmap", data=buf.getvalue(),
                           file_name="difference_heatmap.png", mime="image/png")

    except Exception as e:
        st.error(f"❌ Error processing files: {str(e)}")