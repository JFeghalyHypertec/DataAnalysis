import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from io import BytesIO
from pathlib import Path
import uuid

# Constants
START_ROW = 3  # Row index where actual data starts in the input files
CPU_TIME_SERIE_PLOT = ["CPU Package", "Core Max"]  # Parameters to plot as time series
CORE_LIST = [f"Core {i}" for i in range(26)]       # Expected core column names
DEFAULT_TIME_RANGE_MIN = 5                         # Default smoothing range in minutes


def plot_time_series(time, temperature, label, file_name, time_range_min=DEFAULT_TIME_RANGE_MIN):
    """
    Plot a time series graph for a given parameter (e.g., CPU Package, Core Max).
    - Applies optional smoothing based on the selected time range.
    - Provides download button for saving the plot as PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, temperature, marker='o')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Time Series of {label} in file {file_name}\n Smoothing time: {int(time_range_min)} minutes")
    ax.grid(True)

    # Render plot in Streamlit
    st.pyplot(fig)

    # Save figure to buffer for download
    buf = BytesIO()
    fig.savefig(buf, format="png")

    # Download button with unique key to avoid collisions
    st.download_button(
        label=f"📥 Download {label} Time Series",
        data=buf.getvalue(),
        file_name=f"{file_name}_{label.replace(' ', '_')}_timeseries.png",
        mime="image/png",
        key=f"{file_name}_{label}_download_{uuid.uuid4()}"
    )

    plt.close(fig)


def get_time_series(df, parameter, file_name, time_range_seconds=DEFAULT_TIME_RANGE_MIN * 60):
    """
    Extract and process a time series for a specific parameter.
    - Bins time into user-defined intervals for smoothing.
    - Converts seconds to hours for better readability.
    - Calls plot_time_series() to display and export results.
    """
    for col in df.columns:
        if df.iloc[1, col] == parameter:  # Match the correct column by header
            # Extract raw data (time and parameter values)
            time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
            temperature = pd.to_numeric(df.iloc[START_ROW:, col], errors='coerce')

            # Build dataframe and clean up data
            data = pd.DataFrame({'time': time, 'value': temperature}).dropna().iloc[2:]
            data = data[data['value'] != 0]  # Remove zero values

            # Apply averaging (grouping by time bins)
            if time_range_seconds > 0:
                data['minute_bin'] = (data['time'] // time_range_seconds).astype(int)
                averaged_data = data.groupby('minute_bin').agg({
                    'time': 'mean',
                    'value': 'mean'
                }).reset_index(drop=True)
            else:
                averaged_data = data.copy()

            # Convert time from seconds to hours
            averaged_data['time'] = averaged_data['time'] / 3600

            # Ensure sorted by time before plotting
            averaged_data = averaged_data.sort_values(by='time')

            # Generate plot
            plot_time_series(
                averaged_data['time'], 
                averaged_data['value'], 
                parameter, 
                file_name,
                time_range_min=time_range_seconds/60
            )


def plot_core_temperature_dominance(df, file_name):
    """
    Create a histogram showing how often each core had the maximum temperature.
    - Iterates over rows, finds the hottest core per row, and counts occurrences.
    - Displays as a bar chart and allows download as PNG.
    """
    # Map each core name to its column index if present
    core_indices = {df.iloc[1, i]: i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST}
    if not core_indices:
        st.warning(f"No core columns found in {file_name}.")
        return

    # Initialize counts for each core
    core_counts = {core: 0 for core in CORE_LIST}

    # Iterate over rows to find which core was hottest
    for row_idx in range(START_ROW, df.shape[0]):
        temps = {}
        for core, col_idx in core_indices.items():
            value = pd.to_numeric(df.iloc[row_idx, col_idx], errors='coerce')
            if not np.isnan(value):
                temps[core] = value

        if temps:
            max_temp = max(temps.values())
            # Increment count for the core that had the max temperature
            for core in CORE_LIST:
                if temps.get(core) == max_temp:
                    core_counts[core] += 1
                    break

    # Filter out cores that were never the hottest
    filtered_counts = {k: v for k, v in core_counts.items() if v > 0}
    if not filtered_counts:
        st.warning("No valid temperature data to display dominance.")
        return

    labels = list(filtered_counts.keys())
    counts = list(filtered_counts.values())

    # Create bar plot
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    bars = ax1.bar(labels, counts)
    
    # Add text labels above bars
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=9)
        
    ax1.set_xticklabels(labels, rotation=90)
    ax1.set_ylabel("Times Core was Max")
    ax1.set_title(f"Core Max Temperature Count (Histogram) of {file_name}")

    # Render in Streamlit
    st.pyplot(fig1)

    # Export for download
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png")
    st.download_button(
        "📥 Download Histogram", 
        buf1.getvalue(),
        file_name=f"{file_name}_core_histogram.png", 
        mime="image/png"
    )
    plt.close(fig1)
    

def run_graphs_plot():
    """
    Main Streamlit entry point for plotting graphs.
    - Lets user upload files.
    - Generates time series plots for CPU parameters.
    - Generates histogram of core dominance.
    """
    st.title("Graphs Plot")

    # Toggle for smoothing option
    use_averaging = st.checkbox("🧮 Enable Smoothing", value=True)

    # Choose smoothing time if enabled
    if use_averaging:
        time_range_min = st.number_input(
            "⏱️ Smoothing Time Range (minutes)", 
            min_value=1, max_value=60, value=5, step=1
        )
        time_range_sec = time_range_min * 60
    else:
        time_range_sec = 0  # No averaging
        
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files", 
        type=["csv", "xls", "xlsx"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            ext = Path(file_name).suffix.lower()

            # Load into DataFrame depending on extension
            if ext == ".csv":
                df = pd.read_csv(uploaded_file, header=None)
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(uploaded_file, header=None)
            else:
                st.warning(f"Unsupported file type: {file_name}")
                continue

            # Display which file is being processed
            st.subheader(f"Processing File: {file_name}")

            # Generate time series plots for CPU Package & Core Max
            for parameter in CPU_TIME_SERIE_PLOT:
                st.markdown(f"#### Time Series: {parameter}")
                get_time_series(df, parameter, file_name, time_range_sec)
                time.sleep(0.5)  # Small delay to avoid UI overload

            # Generate histogram for core dominance
            st.markdown(f"#### Core Dominance: {file_name}")
            plot_core_temperature_dominance(df, file_name)
            time.sleep(0.5)