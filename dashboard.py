import streamlit as st
from graphs_plot import run_graphs_plot
from coreHeatmapPlot import run_core_heatmap_plot
from core_heatmap_comparaison import run_core_heatmap_comparaison
from excel_CPU_calculations import run_excel_CPU_calculations

st.set_page_config(page_title="CPU Analysis Dashboard", layout="wide")
st.title("🧠 CPU Data Analysis Dashboard")
st.write("Select a tool below to begin:")

# Simulated pop-up logic
if "active_tool" not in st.session_state:
    st.session_state.active_tool = None

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    if st.button("📊 Excel CPU Calculations"):
        st.session_state.active_tool = "Excel"

with col2:
    if st.button("📈 Graphs Plot"):
        st.session_state.active_tool = "Graphs"

with col3:
    if st.button("🌡️ Core Heatmap Plot"):
        st.session_state.active_tool = "Heatmap"

with col4:
    if st.button("🔥 Core Difference Heatmap"):
        st.session_state.active_tool = "Comparison"

st.divider()

# Upload and execute based on selection
if st.session_state.active_tool == "Excel":
    st.subheader("📊 Excel CPU Calculations")
    run_excel_CPU_calculations()

elif st.session_state.active_tool == "Graphs":
    st.subheader("📈 Graphs Plot")
    run_graphs_plot()

elif st.session_state.active_tool == "Heatmap":
    st.subheader("🌡️ Core Heatmap Plot")
    run_core_heatmap_plot()

elif st.session_state.active_tool == "Comparison":
    st.subheader("🔥 Core Difference Heatmap")
    run_core_heatmap_comparaison()
