import streamlit as st
from graphs_plot import run_graphs_plot
from coreHeatmapPlot import run_core_heatmap_plot
from core_heatmap_comparaison import run_core_heatmap_comparaison
from excel_CPU_calculations import run_excel_CPU_calculations
from core_percentage_difference import run_core_percentage_difference
from core_correlation_matrix import run_core_correlation_matrix
from core_clustering import run_core_clustering
from core_positions import run_core_physical_layout
from core_averages import run_display_core_avg_table
from core_rank_distribution import run_core_rank_distribution

st.set_page_config(page_title="CPU Analysis Dashboard", layout="wide")
st.title("🧠 CPU Data Analysis Dashboard")
st.write("Select a tool below to begin please:")

# Simulated pop-up logic
if "active_tool" not in st.session_state:
    st.session_state.active_tool = None

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)
col7, col8, col9 = st.columns(3)
col10, = st.columns(1)

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

with col5:
    if st.button("📉 Percentage difference"):
        st.session_state.active_tool = "Percentage Difference"
        
with col6:
    if st.button("🧩 Core Correlation Matrix"):
        st.session_state.active_tool = "Correlation Matrix"

with col7:
    if st.button("🌀 Core Clustering"):
        st.session_state.active_tool = "Clustering"

with col8:
    if st.button("📐 Core Physical Layout"):
        st.session_state.active_tool = "Physical Layout"
        
with col9:
    if st.button("🌡️ Average Temperature Table"):
        st.session_state.active_tool = "Average Table"
        
with col10:
    if st.button(" Core Rank Distribution"):
        st.session_state.active_tool = "Core Rank Distribution"

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

elif st.session_state.active_tool == "Percentage Difference":
    st.subheader("📉 Percentage Difference")
    run_core_percentage_difference()
    
elif st.session_state.active_tool == "Correlation Matrix":
    st.subheader("🧩 Core Correlation Matrix")
    run_core_correlation_matrix()
    
elif st.session_state.active_tool == "Clustering":
    st.subheader("🌀 Core Clustering")
    run_core_clustering()
    
elif st.session_state.active_tool == "Physical Layout":
    st.subheader("📐 Core Physical Layout")
    run_core_physical_layout()

elif st.session_state.active_tool == "Average Table":
    st.subheader("🌡️ Average Temperature Table")
    run_display_core_avg_table()
    
elif st.session_state.active_tool == "Core Rank Distribution":
    st.subheader("Core Rank Distribution")
    run_core_rank_distribution()