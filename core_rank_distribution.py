# streamlit_core_rank_distribution.py (Enhanced with Stability Metrics + Automated Report Summary)
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from coreHeatmapPlot import extract_core_data  # your existing extractor

def run_core_rank_distribution():
    st.set_page_config(page_title="Core Rank Distribution", layout="wide")
    st.title("📊 Core-by-Core Rank Distribution")

    # 1) file upload for two sets
    uploaded_files_1 = st.file_uploader(
        "Upload one or more CPU test files (Set 1)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        key="set1"
    )
    label1 = st.text_input("Label for Set 1:", key="label1").strip() or "Set 1"
    
    uploaded_files_2 = st.file_uploader(
        "Upload one or more CPU test files (Set 2)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        key="set2"
    )
    label2 = st.text_input("Label for Set 2:", key="label2").strip() or "Set 2"
    if not uploaded_files_1:
        st.info("Please upload at least one file for Set 1 to get started.")
        st.stop()

    # 2) helper to process files into (name, averages) pairs
    def process_files(uploaded):
        results = []
        for up in uploaded:
            try:
                df = pd.read_excel(up, header=None) if up.name.lower().endswith((".xls", ".xlsx")) else pd.read_csv(up, header=None)
                core_df = extract_core_data(df)
                averages = core_df[core_df != 0].mean()
                results.append((up.name, averages))
            except Exception as e:
                st.error(f"Error processing {up.name}: {e}")
        return results

    results_1 = process_files(uploaded_files_1)
    results_2 = process_files(uploaded_files_2) if uploaded_files_2 else []

    if not results_1:
        st.error("No valid data extracted for Set 1.")
        st.stop()

    # 3) build DataFrames and rank them (1 = hottest)
    df_avgs_1 = pd.DataFrame({fn: avgs for fn, avgs in results_1}).T
    df_ranks_1 = df_avgs_1.rank(axis=1, method="average", ascending=False)
    if results_2:
        df_avgs_2 = pd.DataFrame({fn: avgs for fn, avgs in results_2}).T
        df_ranks_2 = df_avgs_2.rank(axis=1, method="average", ascending=False)

    # 3.1) compute stability metrics
    rank_mean_1 = df_ranks_1.mean(axis=0)
    rank_std_1  = df_ranks_1.std(axis=0)

    # display stability summary for Set 1
    st.subheader(f"🔍 Stability Summary – {label1}")
    summary_1 = pd.DataFrame({"Mean Rank": rank_mean_1, "Rank Std Dev": rank_std_1}).round(2)
    st.dataframe(summary_1)
    most_stable_1   = rank_std_1.idxmin()
    most_variable_1 = rank_std_1.idxmax()
    st.write(f"🔥 **{most_stable_1}** is the most stable core in {label1} (std = {rank_std_1.min():.2f}).")
    st.write(f"⚡ **{most_variable_1}** is the most variable core in {label1} (std = {rank_std_1.max():.2f}).")
    fig_stab1, ax_stab1 = plt.subplots(figsize=(10, 4))
    summary_1['Rank Std Dev'].plot(kind='bar', ax=ax_stab1)
    ax_stab1.set_ylabel('Rank Std Dev')
    ax_stab1.set_title(f'Core Rank Stability – {label1}')
    for i, v in enumerate(summary_1['Rank Std Dev']):
        ax_stab1.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_stab1)

    # repeat stability for Set 2
    if results_2:
        rank_mean_2 = df_ranks_2.mean(axis=0)
        rank_std_2  = df_ranks_2.std(axis=0)
        st.subheader(f"🔍 Stability Summary – {label2}")
        summary_2 = pd.DataFrame({"Mean Rank": rank_mean_2, "Rank Std Dev": rank_std_2}).round(2)
        st.dataframe(summary_2)
        most_stable_2   = rank_std_2.idxmin()
        most_variable_2 = rank_std_2.idxmax()
        st.write(f"🔥 **{most_stable_2}** is the most stable core in {label2} (std = {rank_std_2.min():.2f}).")
        st.write(f"⚡ **{most_variable_2}** is the most variable core in {label2} (std = {rank_std_2.max():.2f}).")
        fig_stab2, ax_stab2 = plt.subplots(figsize=(10, 4))
        summary_2['Rank Std Dev'].plot(kind='bar', ax=ax_stab2)
        ax_stab2.set_ylabel('Rank Std Dev')
        ax_stab2.set_title(f'Core Rank Stability – {label2}')
        for i, v in enumerate(summary_2['Rank Std Dev']):
            ax_stab2.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_stab2)

    # 4) detailed rank distribution selection
    core_names = sorted(df_ranks_1.columns, key=lambda c: int(c.split()[1]))
    core_num = st.number_input(f"Select core (0–{len(core_names)-1})", 0, len(core_names)-1)
    core_name = f"Core {core_num}"
    st.write(f"Distribution for **{core_name}**: {label1} ({len(results_1)} tests)" + (results_2 and f", {label2} ({len(results_2)} tests)" or ""))

    # 5) compute and plot distribution
    counts1 = df_ranks_1[core_name].round().value_counts().sort_index()
    counts2 = results_2 and df_ranks_2[core_name].round().value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = counts1.index
    width = 0.4
    bars1 = ax.bar(pos - width/2, counts1.values, width, label=label1)
    if results_2:
        bars2 = ax.bar(pos + width/2, counts2.reindex(pos, fill_value=0), width, label=label2)
    ax.set_xticks(pos)
    ax.set_xlabel("Rank (1=Hottest)")
    ax.set_ylabel("Test Count")
    ax.set_title(f"{core_name} Rank Histogram")
    ax.legend()
    for bar in bars1 + (bars2 if results_2 else []):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{int(h)}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    # 6) Automated report summary
    st.header("📋 Automated Report Summary")
    report = []
    report.append(f"**{label1}**: {len(results_1)} tests. Stable core: {most_stable_1} (std={rank_std_1.min():.2f}). Variable core: {most_variable_1} (std={rank_std_1.max():.2f}).")
    hot1, cold1 = rank_mean_1.idxmin(), rank_mean_1.idxmax()
    report.append(f"Hottest on average: {hot1} (mean rank={rank_mean_1[hot1]:.2f}). Coldest on average: {cold1} (mean rank={rank_mean_1[cold1]:.2f}).")
    if results_2:
        report.append(f"**{label2}**: {len(results_2)} tests. Stable core: {most_stable_2} (std={rank_std_2.min():.2f}). Variable core: {most_variable_2} (std={rank_std_2.max():.2f}).")
        hot2, cold2 = rank_mean_2.idxmin(), rank_mean_2.idxmax()
        report.append(f"Hottest on average: {hot2} (mean rank={rank_mean_2[hot2]:.2f}). Coldest on average: {cold2} (mean rank={rank_mean_2[cold2]:.2f}).")
    for line in report:
        st.write(line)

    # 7) Download summary
    if st.button("Download Summary as CSV"):
        df_report = pd.DataFrame(report, columns=["Summary"])
        csv = df_report.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="core_rank_summary.csv", mime="text/csv")
