# streamlit_core_rank_distribution.py (Enhanced with Stability Metrics + Multi-Set Comparison + Detailed Distribution with Controls)
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import random

from coreHeatmapPlot import extract_core_data  # your existing extractor

def run_core_rank_distribution():
    st.set_page_config(page_title="Core Rank Distribution", layout="wide")
    st.title("📊 Core-by-Core Rank Distribution")

    # 1) ask how many sets the user wants
    n_sets = st.number_input("Number of test sets:", min_value=1, value=2, step=1)
    sets = []
    for i in range(1, n_sets + 1):
        with st.expander(f"Set {i} Settings", expanded=(i == 1)):
            uploaded = st.file_uploader(
                f"Upload CSV/XLSX files for Set {i}",
                type=["csv", "xls", "xlsx"],
                accept_multiple_files=True,
                key=f"set_{i}"
            )
            label = st.text_input(f"Label for Set {i}:", value=f"Set {i}", key=f"label_{i}").strip() or f"Set {i}"
            sets.append({"label": label, "files": uploaded})

    # require at least first set
    if not sets[0]["files"]:
        st.info("Please upload at least one file for Set 1.")
        st.stop()

    def process_files(uploaded_files):
        results = []
        for up in uploaded_files:
            try:
                df = pd.read_excel(up, header=None) if up.name.lower().endswith((".xls", ".xlsx")) else pd.read_csv(up, header=None)
                core_df = extract_core_data(df)
                avgs = core_df.replace(0, pd.NA).mean()
                results.append((up.name, avgs))
            except Exception as e:
                st.error(f"Error processing {up.name}: {e}")
        return results

    # process each set
    processed = []
    for s in sets:
        res = process_files(s["files"]) if s["files"] else []
        processed.append({"label": s["label"], "results": res})

    # validate first set
    if not processed[0]["results"]:
        st.error("No valid data extracted for Set 1.")
        st.stop()

    # derive core names and max index
    core_names = sorted(
        processed[0]["results"][0][1].index,
        key=lambda c: int(c.split()[1])
    )
    max_core = len(core_names) - 1

    combined_std = {}

    # individual stability summaries
    for item in processed:
        label = item["label"]
        results = item["results"]
        if not results:
            continue

        df_avgs = pd.DataFrame({fn: avgs for fn, avgs in results}).T
        df_ranks = df_avgs.rank(axis=1, method="average", ascending=False)
        rank_mean = df_ranks.mean(axis=0)
        rank_std = df_ranks.std(axis=0)

        # display summary table
        st.subheader(f"🔍 Stability Summary – {label}")
        summary = pd.DataFrame({"Mean Rank": rank_mean, "Rank Std Dev": rank_std}).loc[core_names].round(2)
        st.dataframe(summary)

        stable = rank_std.idxmin()
        variable = rank_std.idxmax()
        st.write(f"🔥 **{stable}** is the most stable core in {label} (std = {rank_std.min():.2f}).")
        st.write(f"⚡ **{variable}** is the most variable core in {label} (std = {rank_std.max():.2f}).")

        # bar chart of std dev
        fig, ax = plt.subplots(figsize=(10, 4))
        summary['Rank Std Dev'].plot(kind="bar", ax=ax)
        ax.set_ylabel("Rank Std Dev")
        ax.set_title(f"Core Rank Stability – {label}")
        for i, v in enumerate(summary['Rank Std Dev']):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        combined_std[label] = rank_std
    used_colors = set()
    # combined comparison plot
    if combined_std:
        st.subheader("📊 Combined Core Rank Stability Comparison")
        combined_df = pd.DataFrame(combined_std).loc[core_names]
        fig, ax = plt.subplots(figsize=(12, 6))
        x = list(range(len(core_names)))
        for label in combined_df.columns:
            y = combined_df[label].values
            while color in used_colors:
                color = "#%06x" % random.randint(0, 0xFFFFFF)
            used_colors.add(color)
            ax.plot(x, y, marker="o", label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(core_names, rotation=90)
        ax.set_ylabel("Rank Std Dev")
        ax.set_title("Core Rank Stability Across Sets")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # detailed core distribution across selected sets
    st.subheader("🔎 Detailed Core Distribution")
    core_num = st.number_input(
        f"Select core number to view (0–{max_core}):", min_value=0, max_value=max_core, value=0, step=1
    )
    core_name = f"Core {core_num}"
    st.write(f"Rank distribution for **{core_name}** across selected sets")

    # choose which sets to display
    available = [item["label"] for item in processed if item["results"]]
    display = st.multiselect("Select sets to display:", options=available, default=available)

    # prepare plot
    positions = list(range(1, len(core_names) + 1))
    n = len(display)
    width = 0.8 / n if n > 0 else 0.8
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, item in enumerate(processed):
        label = item["label"]
        if label not in display or not item["results"]:
            continue
        df_avgs = pd.DataFrame({fn: avgs for fn, avgs in item["results"]}).T
        df_ranks = df_avgs.rank(axis=1, method="average", ascending=False)
        ranks = df_ranks[core_name].round().astype(int)
        counts = ranks.value_counts().reindex(positions, fill_value=0)
        offsets = [p - 0.4 + idx * width for p in positions]
        bars = ax.bar(offsets, counts.values, width=width, label=label)
        # add value labels
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h, f"{int(h)}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(positions)
    ax.set_xlabel("Rank Position (1 = hottest)")
    ax.set_ylabel("Number of Tests")
    ax.set_title(f"{core_name} Rank Distribution")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)