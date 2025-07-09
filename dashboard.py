import streamlit as st # type: ignore
from graphs_plot import run_graphs_plot
from excel_CPU_calculations import run_excel_calculations
from coreHeatmapPlot import run_core_heatmap
from core_heatmap_comparaison import run_heatmap_comparison

st.title("🧠 CPU Data Analysis Dashboard")

option = st.selectbox("Choose an analysis tool:", [
    "📊 Graphs Plot",
    "📈 Excel CPU Calculations",
    "🌡️ Core Heatmap",
    "🔍 Heatmap Comparison"
])

if option == "📊 Graphs Plot":
    run_graphs_plot()
elif option == "📈 Excel CPU Calculations":
    run_excel_calculations()
elif option == "🌡️ Core Heatmap":
    run_core_heatmap()
elif option == "🔍 Heatmap Comparison":
    run_heatmap_comparison()
