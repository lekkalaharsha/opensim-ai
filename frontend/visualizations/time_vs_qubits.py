"""Plot total simulation time vs. number of qubits, grouped by backend."""

import pandas as pd
import plotly.express as px


def plot_time_vs_qubits(df: pd.DataFrame):
    """Return a Plotly figure of total_time vs n_qubits, colored by backend.

    Args:
        df: DataFrame with columns 'n_qubits', 'total_time_seconds', 'backend_name'.

    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.line(
        df.groupby(["n_qubits", "backend_name"])["total_time_seconds"]
        .mean()
        .reset_index(),
        x="n_qubits",
        y="total_time_seconds",
        color="backend_name",
        markers=True,
        title="Average Total Time vs. Qubits",
        labels={"total_time_seconds": "Total Time (s)", "n_qubits": "Number of Qubits"},
    )
    fig.update_layout(legend_title_text="Backend")
    return fig