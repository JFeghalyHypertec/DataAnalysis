# streamlit_core_rank_distribution.py
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from coreHeatmapPlot import extract_core_data  # your existing extractor
def run_spearman_rank_similarity():
    st.set_page_config(page_title="Core Rank Distribution", layout="wide")
    st.title("📊 Core-by-Core Rank Distribution")

    # 1) file upload
    uploaded_files = st.file_uploader(
        "Upload one or more CPU test files (CSV or XLSX)",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload at least one test file to get started.")
        st.stop()

    # 2) process each file: extract averages
    results = []
    for up in uploaded_files:
        try:
            if up.name.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(up, header=None)
            else:
                df = pd.read_csv(up, header=None)

            core_df = extract_core_data(df)
            averages = core_df[core_df != 0].mean()
            results.append((up.name, averages))
        except Exception as e:
            st.error(f"❌ Error processing **{up.name}**: {e}")

    if len(results) == 0:
        st.error("No valid cores data could be extracted.")
        st.stop()

    # 3) build avg DataFrame & compute ranks
    df_avgs = pd.DataFrame({fn: avgs for fn, avgs in results}).T
    df_ranks = df_avgs.rank(axis=1, method="average", ascending=False)

    # 4) let user pick a core number
    #    determine how many cores we actually have (should be 26)
    core_names = sorted(df_ranks.columns, key=lambda c: int(c.split()[1]))
    max_core = len(core_names) - 1

    core_num = st.number_input(
        "Enter core number to display (0–{0})".format(max_core),
        min_value=0,
        max_value=max_core,
        value=0,
        step=1
    )
    core_name = f"Core {int(core_num)}"

    st.write(f"Showing rank distribution for **{core_name}** across {len(results)} tests.")

    # 5) compute how often this core was each rank
    #    round in case of ties
    ranks = df_ranks[core_name].round().astype(int)
    rank_counts = (
        ranks
        .value_counts()
        .reindex(range(1, len(core_names) + 1), fill_value=0)
    )

    # 6) plot bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rank_counts.index, rank_counts.values)
    ax.set_xticks(rank_counts.index)
    ax.set_xlabel("Rank Position (1 = hottest)")
    ax.set_ylabel("Number of Tests")
    ax.set_title(f"{core_name} Rank Occurrence Histogram")
    plt.tight_layout()

    st.pyplot(fig)
    if st.button("Download Rank Distribution Chart"):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
        st.download_button(
            label="Download Chart",
            data=buf.getvalue(),
            file_name=f"{core_name}_rank_distribution.png",
            mime="image/png"
        )