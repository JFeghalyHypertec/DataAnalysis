import streamlit as st
from graphs_plot import run_graphs_plot
from coreHeatmapPlot import run_core_heatmap_plot
from core_heatmap_comparaison import run_core_heatmap_comparaison
from excel_CPU_calculations import run_excel_CPU_calculations

st.set_page_config(page_title="CPU Analysis Dashboard", layout="wide")
st.title("CPU Data Analysis Dashboard")
st.write("Choose a tool and upload file(s) to begin analysis.")

tool = st.selectbox(
    "Select a tool",
    (
        "Excel CPU Calculations",
        "Graphs Plot",
        "Core Heatmap Plot",
        "Core Heatmap Comparison"
    )
)

if tool == "Excel CPU Calculations":
    uploaded_file = st.file_uploader("Upload a CPU data file", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        run_excel_CPU_calculations()

elif tool == "Graphs Plot":
    uploaded_file = st.file_uploader("Upload a CPU data file", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        run_graphs_plot(uploaded_file)

elif tool == "Core Heatmap Plot":
    uploaded_file = st.file_uploader("Upload a CPU data file", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        run_core_heatmap_plot(uploaded_file)

elif tool == "Core Heatmap Comparison":
    file1 = st.file_uploader("Upload the FIRST CPU data file", type=["csv", "xls", "xlsx"], key="file1")
    file2 = st.file_uploader("Upload the SECOND CPU data file", type=["csv", "xls", "xlsx"], key="file2")
    if file1 and file2:
        run_core_heatmap_comparaison(file1, file2)
