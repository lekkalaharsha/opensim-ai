"""Backend comparison bar charts (success rate, average time)."""

import pandas as pd
import plotly.express as px


def plot_success_rate(df: pd.DataFrame):
    """Bar chart of success rate per backend."""
    success_rate = df.groupby("backend_name")["success"].mean().reset_index()
    success_rate["success"] = success_rate["success"] * 100
    fig = px.bar(
        success_rate,
        x="backend_name",
        y="success",
        title="Success Rate by Backend (%)",
        labels={"success": "Success Rate (%)", "backend_name": "Backend"},
        color="backend_name",
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_avg_time(df: pd.DataFrame):
    """Bar chart of average total time per backend."""
    avg_time = df.groupby("backend_name")["total_time_seconds"].mean().reset_index()
    fig = px.bar(
        avg_time,
        x="backend_name",
        y="total_time_seconds",
        title="Average Total Time by Backend",
        labels={"total_time_seconds": "Avg Total Time (s)", "backend_name": "Backend"},
        color="backend_name",
    )
    fig.update_layout(showlegend=False)
    return fig