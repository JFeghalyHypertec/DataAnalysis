# streamlit_core_rank_distribution.py
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
    uploaded_files_2 = st.file_uploader(
        "Upload one or more CPU test files (Set 2)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        key="set2"
    )

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

    # 3) build DataFrames and rank
    df_avgs_1 = pd.DataFrame({fn: avgs for fn, avgs in results_1}).T
    df_ranks_1 = df_avgs_1.rank(axis=1, method="average", ascending=False)
    if results_2:
        df_avgs_2 = pd.DataFrame({fn: avgs for fn, avgs in results_2}).T
        df_ranks_2 = df_avgs_2.rank(axis=1, method="average", ascending=False)

    # 4) choose core number
    core_names = sorted(df_ranks_1.columns, key=lambda c: int(c.split()[1]))
    max_core = len(core_names) - 1
    core_num = st.number_input(
        f"Enter core number to display (0–{max_core})",
        min_value=0, max_value=max_core, value=0, step=1
    )
    core_name = f"Core {core_num}"
    st.write(f"Distribution for **{core_name}**: Set1 = {len(results_1)} tests" + (f", Set2 = {len(results_2)} tests" if results_2 else ""))

    # 5) frequency counts for each rank position
    ranks1 = df_ranks_1[core_name].round().astype(int)
    counts1 = ranks1.value_counts().reindex(range(1, len(core_names)+1), fill_value=0)
    if results_2:
        ranks2 = df_ranks_2[core_name].round().astype(int)
        counts2 = ranks2.value_counts().reindex(range(1, len(core_names)+1), fill_value=0)

    # 6) plot grouped bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = list(range(1, len(core_names)+1))
    width = 0.4
    ax.bar([p - width/2 for p in positions], counts1.values, width=width, label="Set 1")
    if results_2:
        ax.bar([p + width/2 for p in positions], counts2.values, width=width, color="red", label="Set 2")
    ax.set_xticks(positions)
    ax.set_xlabel("Rank Position (1 = hottest)")
    ax.set_ylabel("Number of Tests")
    ax.set_title(f"{core_name} Rank Occurrence Histogram")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # 7) download option
    if st.button("Download Chart"):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        st.download_button(
            "Download as PNG",
            data=buf.getvalue(),
            file_name=f"{core_name}_distribution.png",
            mime="image/png"
        )

if __name__ == "__main__":
    run_core_rank_distribution()
