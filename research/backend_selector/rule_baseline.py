"""Rule-based baseline for backend selection.

A simple heuristic that always picks statevector for small circuits
and MPS for larger ones. Serves as the simplest baseline to beat
with ML models.
"""

import pandas as pd
from typing import Optional


# Thresholds derived from preliminary benchmarks on Tesla T4 (15 GB)
QUBIT_THRESHOLD = 16          # qubits <= 16 → statevector
MEMORY_THRESHOLD_MB = 14000   # if predicted memory > 14 GB, fall back to MPS


def rule_baseline_predict(n_qubits: int) -> str:
    """Predict backend based solely on qubit count.

    Args:
        n_qubits: Number of qubits in the circuit.

    Returns:
        'aer_statevector' or 'aer_mps'.
    """
    if n_qubits <= QUBIT_THRESHOLD:
        return "aer_statevector"
    else:
        return "aer_mps"


def evaluate_baseline(df: pd.DataFrame) -> dict:
    """Evaluate the rule baseline against a dataset of benchmark results.

    Args:
        df: DataFrame with at least columns 'n_qubits', 'backend_name',
            'total_time_seconds', 'success'.

    Returns:
        Dictionary with average regret and accuracy metrics.
    """
    # For each unique circuit (identified by circuit_name, n_qubits, depth)
    # determine the optimal backend from the data
    circuit_groups = df[df["success"]].groupby(["circuit_name", "n_qubits", "depth"])
    optimal = {}
    for (cname, n, d), group in circuit_groups:
        best_row = group.loc[group["total_time_seconds"].idxmin()]
        optimal[(cname, n, d)] = best_row["backend_name"]

    regrets = []
    correct = 0
    total = 0
    for (cname, n, d), opt_backend in optimal.items():
        pred = rule_baseline_predict(n)
        # get the time of the predicted backend
        group = df[(df["circuit_name"] == cname) & (df["n_qubits"] == n) & (df["depth"] == d) & df["success"]]
        pred_time = group[group["backend_name"] == pred]["total_time_seconds"]
        if pred_time.empty:
            continue  # predicted backend didn't succeed, skip
        best_time = group["total_time_seconds"].min()
        regret = pred_time.values[0] / best_time if best_time > 0 else float("inf")
        regrets.append(regret)
        if pred == opt_backend:
            correct += 1
        total += 1

    avg_regret = sum(regrets) / len(regrets) if regrets else float("inf")
    accuracy = correct / total if total > 0 else 0.0
    return {"average_regret": avg_regret, "accuracy": accuracy, "evaluated_circuits": total}