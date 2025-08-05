# streamlit_core_rank_distribution.py (Enhanced with Stability Metrics)
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
                # read CSV or Excel
                df = pd.read_excel(up, header=None) if up.name.lower().endswith((".xls", ".xlsx")) else pd.read_csv(up, header=None)
                core_df = extract_core_data(df)
                # compute per-core average temperature
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

    # 3.1) compute stability metrics: mean rank & std dev per core
    rank_mean_1 = df_ranks_1.mean(axis=0)
    rank_std_1  = df_ranks_1.std(axis=0)

    # display stability summary for Set 1
    st.subheader(f"🔍 Stability Summary – {label1}")
    summary_1 = pd.DataFrame({
        "Mean Rank": rank_mean_1,
        "Rank Std Dev": rank_std_1
    }).round(2)
    st.dataframe(summary_1)

    # highlight most stable and most variable cores
    most_stable_1   = rank_std_1.idxmin()
    most_variable_1 = rank_std_1.idxmax()
    st.write(f"🔥 **{most_stable_1}** is the most stable core in {label1} (std = {rank_std_1.min():.2f}).")
    st.write(f"⚡ **{most_variable_1}** is the most variable core in {label1} (std = {rank_std_1.max():.2f}).")

    # bar chart of rank std dev for Set 1
    fig_stab1, ax_stab1 = plt.subplots(figsize=(10, 4))
    summary_1['Rank Std Dev'].plot(kind='bar', ax=ax_stab1)
    ax_stab1.set_ylabel('Rank Std Dev')
    ax_stab1.set_title(f'Core Rank Stability – {label1}')
    for i, v in enumerate(summary_1['Rank Std Dev']):
        ax_stab1.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_stab1)

    # repeat stability for Set 2 if present
    if results_2:
        rank_mean_2 = df_ranks_2.mean(axis=0)
        rank_std_2  = df_ranks_2.std(axis=0)

        st.subheader(f"🔍 Stability Summary – {label2}")
        summary_2 = pd.DataFrame({
            "Mean Rank": rank_mean_2,
            "Rank Std Dev": rank_std_2
        }).round(2)
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

    # 4) choose core number for detailed rank distribution
    core_names = sorted(df_ranks_1.columns, key=lambda c: int(c.split()[1]))
    max_core = len(core_names) - 1
    core_num = st.number_input(
        f"Enter core number to display (0–{max_core})",
        min_value=0, max_value=max_core, value=0, step=1
    )
    core_name = f"Core {core_num}"
    st.write(
        f"Distribution for **{core_name}**: {label1} = {len(results_1)} tests" +
        (f", {label2} = {len(results_2)} tests" if results_2 else "")
    )

    # 5) frequency counts for each rank position
    ranks1 = df_ranks_1[core_name].round().astype(int)
    counts1 = ranks1.value_counts().reindex(range(1, len(core_names)+1), fill_value=0)
    if results_2:
        ranks2 = df_ranks_2[core_name].round().astype(int)
        counts2 = ranks2.value_counts().reindex(range(1, len(core_names)+1), fill_value=0)

    # 6) plot grouped bar chart of distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = list(range(1, len(core_names)+1))
    width = 0.4
    bars1 = ax.bar([p - width/2 for p in positions], counts1.values, width=width, label=label1)
    if results_2:
        bars2 = ax.bar([p + width/2 for p in positions], counts2.values, width=width, label=label2)
    ax.set_xticks(positions)
    ax.set_xlabel("Rank Position (1 = hottest)")
    ax.set_ylabel("Number of Tests")
    ax.set_title(f"{core_name} Rank Occurrence Histogram")
    ax.legend()
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=8)
    if results_2:
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    # 7) download option for distribution chart
    if st.button("Download Distribution Chart"):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button(
            "Download as PNG",
            data=buf.getvalue(),
            file_name=f"{core_name}_distribution.png",
            mime="image/png"
        )