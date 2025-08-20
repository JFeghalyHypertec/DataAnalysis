# streamlit_core_rank_distribution.py (Enhanced with Stability Metrics + Multi-Set Comparison + AI-Generated Summary)

from io import BytesIO
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import random  # (currently unused; keep if you'll add sampling or demo data later)
import openai
import hashlib
import matplotlib as mpl

# NOTE: expects coreHeatmapPlot.py in PYTHONPATH with an `extract_core_data(df)` function
from coreHeatmapPlot import extract_core_data  # your core extractor

# -----------------------------
# Utilities
# -----------------------------
def color_for(label: str, palette: str = "tab20"):
    """
    Deterministically pick a color for any label by hashing the label
    and indexing into a Matplotlib palette.

    Works with discrete palettes (e.g., tab10/tab20). Falls back to sampling
    a continuous colormap if `.colors` is not available.
    """
    cmap = mpl.cm.get_cmap(palette)
    h = int(hashlib.sha1(str(label).encode("utf-8")).hexdigest(), 16)
    if hasattr(cmap, "colors"):                     # discrete palette (list of RGBA tuples)
        return cmap.colors[h % len(cmap.colors)]
    return cmap((h % 256) / 255.0)                  # continuous palette fallback


# -----------------------------
# Main app
# -----------------------------
def run_core_rank_distribution():
    """
    Streamlit workflow:
    1) Let user define N sets; each set uploads 1..M files and has a label.
    2) For each set:
       - Extract per-core averages across files
       - Compute per-file ranks (higher temp = higher rank)
       - Summarize rank mean and rank stddev (stability proxy)
       - Plot per-core rank std dev bars
    3) Compare rank std dev across sets in one line chart.
    4) Show detailed rank distribution histogram for a selected core across sets.
    5) Produce a brief executive summary + optional AI-generated insights.
    """
    st.title("📊 Core-by-Core Rank Distribution")

    # -------------------------
    # OpenAI API key detection
    # -------------------------
    # Accept from env or Streamlit secrets. If missing, we still run the analytics UI,
    # only the AI summary will be disabled.
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if api_key:
        openai.api_key = api_key
        ai_enabled = True
    else:
        st.info("ℹ️ OpenAI API key not found. Add OPENAI_API_KEY to environment or Streamlit secrets to enable AI summary.")
        ai_enabled = False

    # -------------------------
    # Number of sets & uploads
    # -------------------------
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
            label = st.text_input(
                f"Label for Set {i}:",
                value=f"Set {i}",
                key=f"label_{i}"
            ).strip() or f"Set {i}"
            sets.append({"label": label, "files": uploaded})

    # Require at least one file in the first set
    if not sets[0]["files"]:
        st.info("Please upload at least one file for Set 1.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def process_files(uploaded_files):
        """
        Read each uploaded file, extract per-core data, compute mean per-core
        temperature for that file, and return (filename, Series-of-core-means) tuples.
        """
        out = []
        for up in uploaded_files:
            # Robust file loading for CSV/XLS/XLSX
            df = (
                pd.read_excel(up, header=None)
                if up.name.lower().endswith((".xls", ".xlsx"))
                else pd.read_csv(up, header=None, low_memory=False)
            )
            # Use your shared extractor; must return a df with core columns
            core_df = extract_core_data(df)
            # Replace zeros with NA; compute file-level mean per core
            avgs = core_df.replace(0, pd.NA).mean()
            out.append((up.name, avgs))
        return out

    # -------------------------
    # Process each set
    # -------------------------
    processed = []
    for s in sets:
        res = process_files(s["files"]) if s["files"] else []
        processed.append({"label": s["label"], "results": res})

    # Validate Set 1 has usable results
    if not processed[0]["results"]:
        st.error("No valid data extracted for Set 1.")
        st.stop()

    # Determine core list from first result, preserving numeric order
    core_names = sorted(
        processed[0]["results"][0][1].index,
        key=lambda c: int(c.split()[1])
    )
    max_core = len(core_names) - 1

    combined_std = {}  # collect per-set rank std dev for combined comparison

    # -------------------------
    # Per-set stability summaries
    # -------------------------
    for item in processed:
        label = item["label"]
        results = item["results"]
        if not results:
            continue

        # Build a dataframe: rows = files; columns = cores; values = avg temp
        df_avgs = pd.DataFrame({fn: avgs for fn, avgs in results}).T

        # Rank within each file (row). Highest mean temp -> rank 1 (hottest).
        df_ranks = df_avgs.rank(axis=1, method="average", ascending=False)

        # Per-core mean rank and std dev (stability proxy)
        rank_mean = df_ranks.mean(axis=0)
        rank_std = df_ranks.std(axis=0)

        # Display summary table (ordered by core number)
        st.subheader(f"🔍 Stability Summary – {label}")
        summary = pd.DataFrame({"Mean Rank": rank_mean, "Rank Std Dev": rank_std}).loc[core_names].round(2)
        st.dataframe(summary)

        # Quick callouts
        stable = rank_std.idxmin()
        variable = rank_std.idxmax()
        st.write(f"🔥 **{stable}** is the most stable core in {label} (std = {rank_std.min():.2f}).")
        st.write(f"⚡ **{variable}** is the most variable core in {label} (std = {rank_std.max():.2f}).")

        # Plot bar chart of rank std deviation for this set
        fig, ax = plt.subplots(figsize=(10, 4))
        summary['Rank Std Dev'].plot(kind="bar", ax=ax)
        ax.set_ylabel("Rank Std Dev")
        ax.set_title(f"Core Rank Stability – {label}")
        for i, v in enumerate(summary['Rank Std Dev']):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Store for cross-set comparison
        combined_std[label] = rank_std

    # -------------------------
    # Combined comparison plot
    # -------------------------
    if combined_std:
        st.subheader("📊 Combined Core Rank Stability Comparison")
        combined_df = pd.DataFrame(combined_std).loc[core_names]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = list(range(len(core_names)))
        for label in combined_df.columns:
            y = combined_df[label].values
            ax.plot(x, y, marker="o", label=label, color=color_for(label))
        ax.set_xticks(x)
        ax.set_xticklabels(core_names, rotation=90)
        ax.set_ylabel("Rank Std Dev")
        ax.set_title("Core Rank Stability Across Sets")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------
    # Detailed per-core distribution
    # -------------------------
    st.subheader("🔎 Detailed Core Distribution")
    core_num = st.number_input(
        f"Select core number to view (0–{max_core}):",
        min_value=0, max_value=max_core, value=0, step=1
    )
    core_name = f"Core {core_num}"
    st.write(f"Rank distribution for **{core_name}** across selected sets")

    # Choose which sets to show
    available = [item["label"] for item in processed if item["results"]]
    display = st.multiselect("Select sets to display:", options=available, default=available)

    # Prepare rank position axis (1..N cores)
    positions = list(range(1, len(core_names) + 1))
    n = len(display)
    width = 0.8 / n if n > 0 else 0.8

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, item in enumerate(processed):
        label = item["label"]
        if label not in display or not item["results"]:
            continue

        # Recompute ranks for the selected set
        df_avgs = pd.DataFrame({fn: avgs for fn, avgs in item["results"]}).T
        df_ranks = df_avgs.rank(axis=1, method="average", ascending=False)

        # Pull this core's rank across files, round to nearest integer position
        ranks = df_ranks[core_name].round().astype(int)
        counts = ranks.value_counts().reindex(positions, fill_value=0)

        # Offset bars for grouped display
        offsets = [p - 0.4 + idx * width for p in positions]
        bars = ax.bar(offsets, counts.values, width=width, label=label)

        # Value labels above bars
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h, f"{int(h)}",
                        ha='center', va='bottom', fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(positions)
    ax.set_xlabel("Rank Position (1 = hottest)")
    ax.set_ylabel("Number of Tests")
    ax.set_title(f"{core_name} Rank Distribution")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------
    # Executive Summary (non-AI)
    # -------------------------
    st.subheader("📈 Executive Summary")
    if combined_std:
        # Compute mean of rank std dev across sets per core (lower = more stable)
        overall_mean_std = pd.DataFrame(combined_std).mean(axis=1)
        overall_stable = overall_mean_std.idxmin()
        overall_variable = overall_mean_std.idxmax()
        st.write(
            f"Across all sets, **{overall_stable}** is the most stable core on average "
            f"(mean std = {overall_mean_std.min():.2f})."
        )
        st.write(
            f"Across all sets, **{overall_variable}** is the most variable core on average "
            f"(mean std = {overall_mean_std.max():.2f})."
        )
    else:
        st.write("Not enough data across multiple sets to generate an executive summary.")

    # -------------------------
    # AI-Generated Insights
    # -------------------------
    st.subheader("🤖 AI-Generated Insights")
    if combined_std and ai_enabled:
        try:
            # Prompt is intentionally short; the quantitative findings are already in the UI.
            prompt = (
                f"Summarize the key findings from a CPU core stability analysis across {len(combined_std)} sets. "
                f"Highlight the most stable and most variable cores, overall trends, and any recommendations for optimizing cooling performance."
            )
            # Chat Completions API call (gpt-4o-mini as requested)
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            ai_summary = response.choices[0].message.content.strip()
        except Exception as e:
            ai_summary = f"Could not generate AI summary: {e}"
        st.write(ai_summary)
    elif combined_std and not ai_enabled:
        st.write("AI summary unavailable (no API key).")
    else:
        st.write("AI summary unavailable due to insufficient data.")