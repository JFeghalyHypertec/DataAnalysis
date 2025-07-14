import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from io import BytesIO
from pathlib import Path
import uuid

START_ROW = 3
CPU_TIME_SERIE_PLOT = ["CPU Package", "Core Max"]
CORE_LIST = [f"Core {i}" for i in range(26)]
DEFAULT_TIME_RANGE_MIN = 5


def plot_time_series(time, temperature, label, file_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, temperature, marker='o')
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Time Series of {label} in file {file_name}")
    ax.grid(True)
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label=f"📥 Download {label} Time Series",
        data=buf.getvalue(),
        file_name=f"{file_name}_{label.replace(' ', '_')}_timeseries.png",
        mime="image/png",
        key=f"{file_name}_{label}_download_{uuid.uuid4()}"  # Unique key to avoid conflicts
    )
    plt.close(fig)

def get_time_series(df, parameter, file_name, time_range_seconds=DEFAULT_TIME_RANGE_MIN * 60):
    for col in df.columns:
        if df.iloc[1, col] == parameter:
            time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
            temperature = pd.to_numeric(df.iloc[START_ROW:, col], errors='coerce')

            # Create raw dataframe
            data = pd.DataFrame({'time': time, 'value': temperature}).dropna().iloc[2:]

            # Remove 0s from value column
            data = data[data['value'] != 0]

            # Bin time into 60-second intervals (as integers)
            if time_range_seconds > 0:
                data['minute_bin'] = (data['time'] // time_range_seconds).astype(int)
                averaged_data = data.groupby('minute_bin').agg({
                    'time': 'mean',
                    'value': 'mean'
                }).reset_index(drop=True)
            else:
                averaged_data = data.copy()


            # Group by each minute and average the temperature
            averaged_data = data.groupby('minute_bin').agg({
                'time': 'mean',
                'value': 'mean'
            }).reset_index(drop=True)

            # Convert time to hours
            averaged_data['time'] = averaged_data['time'] / 3600

            # Sort by time
            averaged_data = averaged_data.sort_values(by='time')

            # Plot
            plot_time_series(averaged_data['time'], averaged_data['value'], parameter, file_name)


def plot_core_temperature_dominance(df, file_name):
    core_indices = {df.iloc[1, i]: i for i in range(df.shape[1]) if df.iloc[1, i] in CORE_LIST}
    if not core_indices:
        st.warning(f"No core columns found in {file_name}.")
        return

    core_counts = {core: 0 for core in CORE_LIST}
    for row_idx in range(START_ROW, df.shape[0]):
        temps = {}
        for core, col_idx in core_indices.items():
            value = pd.to_numeric(df.iloc[row_idx, col_idx], errors='coerce')
            if not np.isnan(value):
                temps[core] = value
        if temps:
            max_temp = max(temps.values())
            for core in CORE_LIST:
                if temps.get(core) == max_temp:
                    core_counts[core] += 1
                    break

    filtered_counts = {k: v for k, v in core_counts.items() if v > 0}
    if not filtered_counts:
        st.warning("No valid temperature data to display dominance.")
        return

    labels = list(filtered_counts.keys())
    counts = list(filtered_counts.values())

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(labels, counts)
    ax1.set_xticklabels(labels, rotation=90)
    ax1.set_ylabel("Times Core was Max")
    ax1.set_title(f"Core Max Temperature Count (Histogram) of {file_name}")
    st.pyplot(fig1)
    
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png")
    st.download_button("📥 Download Histogram", buf1.getvalue(),
                       file_name=f"{file_name}_core_histogram.png", mime="image/png")
    plt.close(fig1)
    
def run_graphs_plot():
    st.title("Graphs Plot")
    use_averaging = st.checkbox("🧮 Enable Averaging", value=True)

    if use_averaging:
        time_range_min = st.number_input("⏱️ Averaging Time Range (minutes)", min_value=1, max_value=60, value=5, step=1)
        time_range_sec = time_range_min * 60
    else:
        time_range_sec = 0  # no averaging
        
    uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            ext = Path(file_name).suffix.lower()

            if ext == ".csv":
                df = pd.read_csv(uploaded_file, header=None)
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(uploaded_file, header=None)
            else:
                st.warning(f"Unsupported file type: {file_name}")
                continue

            st.subheader(f"Processing File: {file_name}")
            for parameter in CPU_TIME_SERIE_PLOT:
                st.markdown(f"#### Time Series: {parameter}")
                get_time_series(df, parameter, file_name, time_range_sec)
                time.sleep(0.5)

            st.markdown(f"#### Core Dominance: {file_name}")
            plot_core_temperature_dominance(df, file_name)
            time.sleep(0.5)

# Uncomment if you want to run standalone
if __name__ == "__main__":
    run_graphs_plot()
