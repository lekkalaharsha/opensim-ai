"""Dataset quality assurance checks for OpenQSim.

Run this after dataset assembly to generate a quality report.
Usage: python benchmark/quality.py --dataset data/datasets/openqsim_v0.1-small
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def load_dataset(dataset_dir: str):
    """Load results.csv and circuits.json."""
    results_path = Path(dataset_dir) / "results.csv"
    circuits_path = Path(dataset_dir) / "circuits.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")
    df = pd.read_csv(results_path)
    return df


def check_timing_positive(df: pd.DataFrame) -> list:
    issues = []
    for col in ["compilation_time_seconds", "execution_time_seconds", "total_time_seconds"]:
        neg = df[df[col] < 0]
        if len(neg) > 0:
            issues.append(f"{len(neg)} records have negative {col}")
    zero_exec = df[(df["execution_time_seconds"] == 0) & (df["success"] == True)]
    if len(zero_exec) > 0:
        issues.append(f"{len(zero_exec)} successful records have zero execution time")
    return issues


def check_memory_plausible(df: pd.DataFrame, max_mb=15360) -> list:
    issues = []
    over = df[df["peak_gpu_memory_mb"] > max_mb]
    if len(over) > 0:
        issues.append(f"{len(over)} records exceed GPU memory ({max_mb} MB)")
    return issues


def check_fidelity_bounds(df: pd.DataFrame) -> list:
    issues = []
    fid = df[df["fidelity"].notna()]
    out_of_range = fid[(fid["fidelity"] < 0) | (fid["fidelity"] > 1)]
    if len(out_of_range) > 0:
        issues.append(f"{len(out_of_range)} records have fidelity outside [0,1]")
    return issues


def check_ghz_fidelity(df: pd.DataFrame) -> list:
    issues = []
    ghz = df[(df["circuit_name"].str.startswith("ghz")) & (df["n_qubits"] <= 16) & df["success"]]
    low_fid = ghz[ghz["fidelity"] < 0.99]
    if len(low_fid) > 0:
        issues.append(f"{len(low_fid)} GHZ runs (≤16 qubits) have fidelity < 0.99")
    return issues


def check_success_error_consistency(df: pd.DataFrame) -> list:
    issues = []
    failed_no_msg = df[(df["success"] == False) & (df["error_message"].isna())]
    if len(failed_no_msg) > 0:
        issues.append(f"{len(failed_no_msg)} failed records lack error_message")
    return issues


def check_ghz_entropy(df: pd.DataFrame) -> list:
    issues = []
    ghz_exact = df[(df["circuit_name"].str.startswith("ghz")) & (df["entropy_method"] == "exact")]
    wrong_ent = ghz_exact[abs(ghz_exact["entropy"] - 1.0) > 0.1]
    if len(wrong_ent) > 0:
        issues.append(f"{len(wrong_ent)} GHZ exact entropy records deviate from 1.0")
    return issues


def run_all_checks(df: pd.DataFrame) -> dict:
    checks = {
        "timing_positive": check_timing_positive(df),
        "memory_plausible": check_memory_plausible(df),
        "fidelity_bounds": check_fidelity_bounds(df),
        "ghz_fidelity": check_ghz_fidelity(df),
        "success_error_consistency": check_success_error_consistency(df),
        "ghz_entropy": check_ghz_entropy(df),
    }
    return checks


def generate_report(dataset_dir: str):
    df = load_dataset(dataset_dir)
    results = run_all_checks(df)

    report_lines = ["# OpenQSim Dataset Quality Report", f"Dataset: {dataset_dir}"]
    total_issues = 0
    for name, issues in results.items():
        report_lines.append(f"\n## {name}")
        if not issues:
            report_lines.append("✅ All good")
        else:
            total_issues += len(issues)
            for issue in issues:
                report_lines.append(f"- ❌ {issue}")

    report_lines.insert(1, f"\nOverall issues found: {total_issues}\n")
    report_path = Path(dataset_dir) / "QUALITY_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    args = parser.parse_args()
    generate_report(args.dataset)