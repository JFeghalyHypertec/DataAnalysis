# core_heatmap_comparaison.py
# Streamlit module: compare two OCCT core-temperature CSVs and show a difference heatmap + per-core avg bars

import io
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import streamlit as st

# -----------------------------
# Config
# -----------------------------
START_ROW = 3  # data starts at row index 3 (0-based), as in your existing scripts
CORE_LIST = [f"Core {i}" for i in range(26)]  # expected core column names in the header row (row index 1)
TIME_COL_INDEX = 0  # first column is time (as in your existing sheets); we normalize to start at 0
EXCLUDE_ZEROS_FROM_AVG = True  # replace 0 with NaN before averaging (optional but common for OCCT exports)

# -----------------------------
# IO / Parsing
# -----------------------------
def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """
    Reads an uploaded OCCT-like CSV with no header row, preserving all rows.
    We keep header-less to let extract_core_data pull names from row index 1.
    """
    # force UTF-8 with fallback
    data = uploaded_file.read()
    try:
        df = pd.read_csv(io.BytesIO(data), header=None, engine="python")
    except Exception:
        df = pd.read_csv(io.BytesIO(data), header=None, engine="python", encoding_errors="ignore")
    return df


def extract_core_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time series of core temperatures.
    - Assumes row index 1 contains the header names (e.g., "CPU Package", "Core 0", ..., "Core 25").
    - Data starts at START_ROW.
    - Returns a DataFrame indexed by time (seconds from 0), columns = common core labels.
    """
    if df.shape[0] <= START_ROW + 1 or df.shape[1] < 2:
        raise ValueError("File format too short or malformed: not enough rows/columns.")

    # find core columns by name in row index 1
    header_row = df.iloc[1, :]
    core_cols = [i for i in range(df.shape[1]) if header_row.iloc[i] in CORE_LIST]

    if not core_cols:
        raise ValueError("No core temperature columns found (expected headers like 'Core 0', 'Core 1', ...).")

    # time vector from first column
    time = pd.to_numeric(df.iloc[START_ROW:, TIME_COL_INDEX], errors="coerce")
    time = time - time.iloc[0]  # normalize to start at 0

    # core data rows
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors="coerce")

    # build result
    out = core_data.copy()
    out.index = time.values
    out.columns = [header_row.iloc[i] for i in core_cols]

    # drop all-NaN rows at the very end
    out = out[~out.isna().all(axis=1)]

    return out


# -----------------------------
# Alignment / Difference
# -----------------------------
def align_common(core1: pd.DataFrame, core2: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
    """
    Returns (common_times, common_cols)
    """
    common_times = core1.index.intersection(core2.index)
    common_cols = core1.columns.intersection(core2.columns)
    return common_times, common_cols


def compute_diff(core1: pd.DataFrame, core2: pd.DataFrame) -> pd.DataFrame:
    """
    Compute difference DataFrame (file2 - file1) for common times and cores.
    Index converted to hours (rounded).
    """
    idx, cols = align_common(core1, core2)

    if len(idx) == 0 or len(cols) == 0:
        return pd.DataFrame()

    a1 = core1.loc[idx, cols]
    a2 = core2.loc[idx, cols]

    df_diff = a2 - a1
    if df_diff.empty:
        return df_diff

    # convert index (seconds) -> hours for nicer axis
    df_diff.index = (pd.to_numeric(df_diff.index, errors="coerce") / 3600.0).round(3)
    return df_diff


# -----------------------------
# Plotting
# -----------------------------
def _safe_xlim_for_avgs(avgs: pd.Series) -> Tuple[float, float]:
    """
    Produce finite axis limits for average bar chart even if avgs has NaN/Inf or is empty.
    """
    if avgs is None or avgs.empty:
        return (-1.0, 1.0)

    vals = avgs.values.astype(float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 1.0)

    xmin = float(vals.min())
    xmax = float(vals.max())
    if xmin == xmax:  # flat case
        pad = max(0.5, abs(xmax) * 0.2 + 0.1)
        return (xmin - pad, xmax + pad)

    pad = max(0.5, 0.1 * max(abs(xmin), abs(xmax)))
    return (xmin - pad, xmax + pad)


def plot_diff_heatmap_and_bars(df_diff: pd.DataFrame, title_left: str, title_right: str) -> None:
    """
    Create a two-pane figure:
      - Left: heatmap (time vs core) of difference
      - Right: horizontal bar chart of per-core average difference
    Handles empty / NaN data gracefully.
    """
    st.subheader("ΔTemp Heatmap (File 2 - File 1) and Per-Core Averages")

    # Early validation
    if df_diff is None or df_diff.empty:
        st.warning("No valid overlap between files (time and/or cores). Unable to compute differences.")
        return

    # Optionally treat zeros as NaN for averaging only
    diff_for_avg = df_diff.copy()
    if EXCLUDE_ZEROS_FROM_AVG:
        diff_for_avg = diff_for_avg.replace(0, np.nan)

    # Compute per-core averages (drop cores that are all-NaN)
    avgs = diff_for_avg.mean(axis=0).dropna()
    avgs = avgs.sort_index(ascending=True)  # ensure Core 0 ... Core n order
    avgs_for_bars = avgs[::-1]  # reverse for a nicer top-down bar list

    # Create figure
    fig = plt.figure(figsize=(14, 7))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1.25, 1], wspace=0.25)

    # ---- Heatmap (left) ----
    ax0 = fig.add_subplot(spec[0, 0])

    # Prepare data matrix for imshow: rows=time, cols=cores
    # Sort by time increasing and columns in natural order
    df_plot = df_diff.sort_index(axis=0)
    df_plot = df_plot.reindex(sorted(df_plot.columns, key=lambda c: int(c.split()[-1])), axis=1)

    if df_plot.shape[0] == 0 or df_plot.shape[1] == 0 or df_plot.to_numpy(allna=True).size == 0:
        ax0.text(0.5, 0.5, "No data to plot heatmap", ha="center", va="center")
        ax0.axis("off")
    else:
        mat = df_plot.to_numpy()
        # Handle all-NaN gracefully by filling for display (cmap will still show neutral if near 0)
        if np.all(~np.isfinite(mat)):
            mat = np.zeros_like(mat)

        im = ax0.imshow(
            mat,
            aspect="auto",
            origin="lower",
            interpolation="nearest"
        )
        cb = fig.colorbar(im, ax=ax0)
        cb.set_label("ΔTemp (°C)")

        # X-axis: cores
        ax0.set_xticks(range(df_plot.shape[1]))
        ax0.set_xticklabels(df_plot.columns, rotation=90)
        # Y-axis: time (hours)
        ax0.set_yticks(np.linspace(0, df_plot.shape[0]-1, min(df_plot.shape[0], 6)).astype(int))
        if df_plot.shape[0] > 0:
            ylabels = df_plot.index.values[
                np.linspace(0, df_plot.shape[0]-1, min(df_plot.shape[0], 6)).astype(int)
            ]
            ax0.set_yticklabels([f"{y:.2f} h" for y in ylabels])
        ax0.set_xlabel("Core")
        ax0.set_ylabel("Time (h)")
        ax0.set_title(title_left)

    # ---- Bar chart (right) ----
    ax1 = fig.add_subplot(spec[0, 1])
    ax1.set_title(title_right)

    if avgs_for_bars.empty:
        # Show an empty panel with safe limits and a message
        ax1.set_xlim(-1, 1)
        ax1.set_xlabel("ΔTemp (°C)")
        ax1.text(0.5, 0.5, "No valid data to compute per-core averages.", ha="center", va="center")
        ax1.set_yticks([])
    else:
        bars = ax1.barh(avgs_for_bars.index, avgs_for_bars.values)
        ax1.set_xlabel("ΔTemp (°C)")

        # Safe, finite x-limits
        xmin, xmax = _safe_xlim_for_avgs(avgs_for_bars)
        ax1.set_xlim(xmin, xmax)

        # Annotate bar values
        span = xmax - xmin
        offset = 0.02 * span
        for b, v in zip(bars, avgs_for_bars.values):
            # place text just beyond the end of the bar
            x = v + (offset if v >= 0 else -offset)
            ax1.text(x, b.get_y() + b.get_height()/2, f"{v:.2f}°C", va="center")

        # Grid for readability
        ax1.grid(axis="x", linestyle="--", alpha=0.4)

    st.pyplot(fig)


# -----------------------------
# Streamlit Entrypoint
# -----------------------------
def run_core_heatmap_comparaison():
    """
    Streamlit UI: upload two CSV files, parse, align, compute (file2 - file1),
    and plot a difference heatmap + per-core average bars with robust guards.
    """
    st.header("Core Heatmap Comparison (File 2 − File 1)")

    st.markdown(
        "Upload **two** OCCT CSV exports with core temperatures. "
        "This tool aligns common timestamps and cores, computes the difference (File 2 − File 1), "
        "and visualizes it as a heatmap with per-core average deltas."
    )

    colA, colB = st.columns(2)
    with colA:
        file1 = st.file_uploader("File 1 (baseline/reference)", type=["csv"], key="cmp_f1")
    with colB:
        file2 = st.file_uploader("File 2 (comparison)", type=["csv"], key="cmp_f2")

    if not file1 or not file2:
        st.info("Please upload both files to proceed.")
        return

    # Parse files
    try:
        df1 = read_uploaded_csv(file1)
        df2 = read_uploaded_csv(file2)
    except Exception as e:
        st.error(f"Failed to read uploaded files: {e}")
        return

    # Extract core data
    try:
        core1 = extract_core_data(df1)
        core2 = extract_core_data(df2)
    except Exception as e:
        st.error(f"Failed to extract core data: {e}")
        return

    if core1.empty or core2.empty:
        st.error("Extracted core data is empty in one or both files.")
        return

    # Quick sanity: ensure we have some overlap in time and columns
    idx, cols = align_common(core1, core2)
    if len(idx) == 0:
        st.warning("No overlapping timestamps between the two files after normalization.")
    if len(cols) == 0:
        st.warning("No common core columns between the two files (e.g., Core 0..N).")

    # Compute difference
    df_diff = compute_diff(core1, core2)

    # Titles
    title_left = "ΔTemp Heatmap (aligned time × core)"
    title_right = "Average ΔTemp per Core"

    # Plot (with all guards built-in)
    plot_diff_heatmap_and_bars(df_diff, title_left, title_right)


# If you want to run this module directly with `streamlit run core_heatmap_comparaison.py`,
# uncomment the lines below.
# if __name__ == "__main__":
#     run_core_heatmap_comparaison()
