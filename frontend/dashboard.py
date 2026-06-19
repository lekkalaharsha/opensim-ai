"""OpenQSim Benchmark Suite – Streamlit Dashboard.

Usage:
    streamlit run frontend/dashboard.py
"""

import sys
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

from frontend.visualizations.time_vs_qubits import plot_time_vs_qubits
from frontend.visualizations.memory_vs_qubits import plot_memory_vs_qubits
from frontend.visualizations.backend_comparison import (
    plot_success_rate,
    plot_avg_time,
)

# ----------------------------------------------------------------------
# Page config
st.set_page_config(
    page_title="OpenQSim Dashboard",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ OpenQSim Benchmark Suite")
st.caption("An open-source dataset and AI-assisted platform for quantum circuit simulation.")

# ----------------------------------------------------------------------
# Sidebar – data loading
st.sidebar.header("Data Source")
data_path = st.sidebar.text_input(
    "Path to results.csv",
    value="data/datasets/openqsim_v0.1-small/results.csv",
)

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        # Ensure success is boolean
        if "success" in df.columns and df["success"].dtype != bool:
            df["success"] = df["success"].astype(bool)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data(data_path)

if df.empty:
    st.warning("No data loaded. Run a benchmark sweep first, or adjust the path.")
    st.stop()

# ----------------------------------------------------------------------
# Main content
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 Performance", "🔬 Backend Comparison", "🤖 Backend Selector"]
)

# ----- Overview tab -----
with tab1:
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Unique Circuits", df["circuit_name"].nunique())
    if "success" in df.columns:
        success_count = df["success"].astype(bool).sum()
    else:
        success_count = 0
    col3.metric("Successful Runs", success_count)
    col4.metric("Backends", ", ".join(sorted(df["backend_name"].unique())))
    st.dataframe(df.head(100), use_container_width=True)

# ----- Performance tab -----
with tab2:
    st.header("Performance Scaling")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_time_vs_qubits(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_memory_vs_qubits(df), use_container_width=True)

# ----- Backend comparison tab -----
with tab3:
    st.header("Backend Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_success_rate(df), use_container_width=True)
    with col2:
        st.plotly_chart(plot_avg_time(df), use_container_width=True)

# ----- Backend Selector (placeholder) -----
with tab4:
    st.header("AI Backend Selector")
    st.info("This feature will be available in Phase 2 (Month 2). It will use XGBoost + NVIDIA NIM to predict the optimal backend.")
    qasm_input = st.text_area("Paste QASM circuit here:", height=200, disabled=True)
    if st.button("Predict Backend", disabled=True):
        st.write("Prediction will appear here.")