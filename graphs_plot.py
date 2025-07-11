import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from io import StringIO
from pathlib import Path

START_ROW = 3

CPU_TIME_SERIE_PLOT = ["CPU Package", "Core Max"]

CORE_LIST = [f"Core {i}" for i in range(26)]

def plot_time_series(time, temperature, label, file_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, temperature, marker='o')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Time Series of {label} in file {file_name}")
    ax.grid(True)
    st.pyplot(fig)

def get_time_series(df, parameter, file_name):
    for col in df.columns:
        if df.iloc[1, col] == parameter:
            time = pd.to_numeric(df.iloc[START_ROW:, 0], errors='coerce')
            temperature = pd.to_numeric(df.iloc[START_ROW:, col], errors='coerce')
            label = df.iloc[1, col]

            plot_data = pd.DataFrame({'time': time, 'value': temperature}).dropna().iloc[2:]
            plot_data = plot_data.sort_values(by='time')
            plot_time_series(plot_data['time'], plot_data['value'], label, file_name)

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
    total = sum(counts)
    percentages = [c / total * 100 for c in counts]

    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(labels, counts)
    ax1.set_xticklabels(labels, rotation=90)
    ax1.set_ylabel("Times Core was Max")
    ax1.set_title(f"Core Max Temperature Count (Histogram) of {file_name}")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.pie(percentages, labels=labels, autopct="%1.1f%%", startangle=90, counterclock=False)
    ax2.set_title(f"Core Max Temperature Percentage Dominance (Pie Chart) of {file_name}")
    ax2.axis("equal")
    st.pyplot(fig2)

def run_graphs_plot():
    st.title("Graphs Plot")
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
                get_time_series(df, parameter, file_name)
                time.sleep(0.5)

            st.markdown(f"#### Core Dominance: {file_name}")
            plot_core_temperature_dominance(df, file_name)
            time.sleep(0.5)

# Uncomment if you want to run standalone
if __name__ == "__main__":
    run_graphs_plot()
