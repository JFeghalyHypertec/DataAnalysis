import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

# -----------------------------
# Constants
# -----------------------------
START_ROW = 3  # Data (time + sensors) starts from this row index
CORE_LIST = [f"Core {i}" for i in range(26)]  # Expected core column headers
MINUTES_SMOOTHING = 15  # Size of the time bucket in minutes


# -----------------------------
# Data extraction
# -----------------------------
def extract_core_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts per-core temperature data from a raw OCCT dataframe.

    - Detects columns for core temps by checking header row (row index 1).
    - Parses the first column as time (seconds).
    - Returns a DataFrame indexed by time with columns = core names.
    """
    # Identify core columns from header row
    core_cols = [i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST]
    if not core_cols:
        raise ValueError("No core temperature columns found.")

    # Time column (first col), numeric
    time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')

    # Core temps to numeric; keep NaN for bad parses
    core_data = df.iloc[START_ROW:, core_cols].apply(pd.to_numeric, errors='coerce')
    core_data.index = time
    core_data.columns = [df.iloc[1, i] for i in core_cols]

    return core_data


# -----------------------------
# Minute-bucket averages
# -----------------------------
def calculate_averages_over_time(core_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes average temperature across cores in MINUTES_SMOOTHING-minute buckets.

    Returns a DataFrame with:
    - 'Time': bucket timestamp in HOURS (float)
    - 'Average': mean temperature of all cores within each bucket
                 (ignoring zeros and NaNs)
    """
    # Work on a copy to avoid in-place SettingWithCopy warnings
    cleaned_data = core_df.replace(0, np.nan).copy()

    # Create minute-level buckets (e.g., 0, 900, 1800 seconds for 15-min buckets)
    cleaned_data['minute'] = (cleaned_data.index // (MINUTES_SMOOTHING * 60)).astype(int) * 60 * MINUTES_SMOOTHING

    # Per-row (per-second) average across cores, NaN-safe
    per_second_avg = cleaned_data.drop(columns='minute').mean(axis=1, skipna=True)

    # Combine time + per-second average
    temp_df = pd.DataFrame({
        'Time': per_second_avg.index,
        'Average': per_second_avg.values,
        'Minute': cleaned_data['minute']
    })

    # Aggregate to bucket average
    result = temp_df.groupby('Minute', as_index=False)['Average'].mean()

    # Rename to 'Time' and convert seconds → hours for plotting
    result.rename(columns={'Minute': 'Time'}, inplace=True)
    result['Time'] = result['Time'] / 3600.0

    return result


# -----------------------------
# % Difference plotting
# -----------------------------
def plot_percentage_difference(averages: pd.DataFrame, file_name: str) -> plt.Figure:
    """
    Plots the percentage difference of average temperature vs. the first bucket.

    %Δ = (avg(t) - avg(t0)) / avg(t0) * 100
    """
    if averages.empty or averages['Average'].isna().all():
        raise ValueError("Averages are empty or all NaN; cannot compute percentage differences.")

    baseline = averages['Average'].iloc[0]

    # Guard against divide-by-zero or NaN baseline
    if pd.isna(baseline) or baseline == 0:
        raise ValueError("Baseline (first bucket average) is NaN or zero; cannot compute percentage differences.")

    # Work on a copy to avoid altering caller’s DataFrame
    avg_plot = averages.copy()
    avg_plot['% Difference'] = ((avg_plot['Average'] - baseline) / baseline) * 100.0

    # Build the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(avg_plot['Time'], avg_plot['% Difference'], marker='o', linestyle='-', color='purple')
    ax.set_title(f"% Difference Over Time for {file_name}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Percentage Difference (%)")
    ax.grid(True)

    # Reference line at 0%
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    return fig


# -----------------------------
# Streamlit app entry
# -----------------------------
def run_core_percentage_difference():
    """
    Streamlit workflow:
    - Upload one or more CSV/XLS/XLSX OCCT files.
    - For each file: extract cores → bucket averages → %Δ vs first bucket → plot + download.
    """
    st.header("📉 Percentage Difference")

    uploaded_files = st.file_uploader(
        "Upload one or more CSV or Excel files",
        type=['csv', 'xls', 'xlsx'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            try:
                # Load file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, header=None)
                else:
                    df = pd.read_excel(uploaded_file, header=None)

                # Extract core temps and compute bucket averages
                core_df = extract_core_data(df)
                averages = calculate_averages_over_time(core_df)

                # Plot % difference vs. baseline
                fig = plot_percentage_difference(averages, file_name)

                st.subheader(f"📊 Percentage Difference for: `{file_name}`")
                st.pyplot(fig)

                # Download button (PNG)
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label=f"💾 Download Percentage Difference for {file_name}",
                    data=buf.getvalue(),
                    file_name=f"{file_name.replace('.','_')}_percentage_difference.png",
                    mime="image/png"
                )
                st.markdown("---")  # Divider between plots

            except Exception as e:
                st.error(f"❗ Error processing file `{file_name}`: {e}")
