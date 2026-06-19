"""Plot peak GPU memory vs. number of qubits, grouped by backend."""

import pandas as pd
import plotly.express as px


def plot_memory_vs_qubits(df: pd.DataFrame):
    """Return a Plotly figure of peak_gpu_memory_mb vs n_qubits, colored by backend.

    Args:
        df: DataFrame with columns 'n_qubits', 'peak_gpu_memory_mb', 'backend_name'.

    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.line(
        df.groupby(["n_qubits", "backend_name"])["peak_gpu_memory_mb"]
        .mean()
        .reset_index(),
        x="n_qubits",
        y="peak_gpu_memory_mb",
        color="backend_name",
        markers=True,
        title="Average Peak GPU Memory vs. Qubits",
        labels={"peak_gpu_memory_mb": "Peak GPU Memory (MB)", "n_qubits": "Number of Qubits"},
    )
    fig.update_layout(legend_title_text="Backend")
    return fig