import streamlit as st
from graphs_plot import run_graphs_plot
from coreHeatmapPlot import run_core_heatmap_plot
from core_heatmap_comparaison import run_core_heatmap_comparaison

from excel_CPU_calculations import run_excel_CPU_calculations

st.set_page_config(page_title="CPU Analysis Dashboard", layout="wide")

st.title("CPU Data Analysis Dashboard")
st.write("Choose one of the tools below to run.")

option = st.selectbox(
    "Select a tool to run",
    (
        "Excel CPU Calculations",
        "Graphs Plot",
        "Core Heatmap Plot",
        "Core Heatmap Comparison"
    )
)

st.warning("This dashboard is built for desktop usage. If nothing happens, please check if a file dialog appeared in the background.")

if st.button("Run Selected Tool"):
    if option == "Excel CPU Calculations":
        run_excel_CPU_calculations()
    elif option == "Graphs Plot":
        run_graphs_plot()
    elif option == "Core Heatmap Plot":
        run_core_heatmap_plot()
    elif option == "Core Heatmap Comparison":
        run_core_heatmap_comparaison()