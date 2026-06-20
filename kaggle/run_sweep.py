"""Main script for running a benchmark sweep on Kaggle.

This script:
1. Parses a YAML sweep configuration file.
2. Iterates through all combinations of circuits, qubits, depths, and backends.
3. Calls the benchmark runner for each combination.
4. Implements robust checkpointing to save raw JSON results and a summary CSV
   to /kaggle/working/, ensuring progress is not lost.
5. Handles per-benchmark errors gracefully so a single failure never aborts
   the whole sweep.
"""

import sys
import os
import yaml
import time
import logging
from pathlib import Path
from itertools import product
import dataclasses

# Ensure the project root is on the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from benchmark.runner import run_single_benchmark, load_circuit_generator, load_backend

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Kaggle-specific paths
KAGGLE_OUTPUT_DIR = Path("/kaggle/working/")
DEFAULT_SWEEP_CONFIG = Path(__file__).parent.parent / "benchmark" / "sweep_config_0a.yaml"


def load_sweep_config(config_path: Path) -> dict:
    """Loads and validates the sweep configuration YAML."""
    logging.info(f"Loading sweep configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Basic validation
    required_keys = ["circuits", "qubits", "depths", "backends", "repetitions", "seed_base"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Sweep config missing required key: '{key}'")
    return config


def results_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Converts a list of SimulationResult-like dicts to a pandas DataFrame."""
    flat_records = []
    for r in records:
        flat = r.copy()
        env = flat.pop("environment", {})
        if env:
            for k, v in env.items():
                flat[f"env_{k}"] = v
        # Remove large objects for CSV summary
        flat.pop("statevector", None)
        flat.pop("circuit_fingerprint", None)
        flat_records.append(flat)
    return pd.DataFrame(flat_records)


def run_sweep(config_path: Path):
    """Executes the full benchmark sweep based on the config file."""
    config = load_sweep_config(config_path)

    circ_names, qubit_counts, depths, backend_names = config["circuits"], config["qubits"], config["depths"], config["backends"]
    repetitions, seed_base = config["repetitions"], config["seed_base"]

    # Create output directories on Kaggle
    raw_json_dir = KAGGLE_OUTPUT_DIR / "raw"
    raw_json_dir.mkdir(exist_ok=True)

    # --- State management for checkpointing ---
    all_results = []
    processed_combinations = set()
    summary_csv_path = KAGGLE_OUTPUT_DIR / "results_summary.csv"

    # Resume from previous checkpoint if it exists
    if summary_csv_path.exists():
        logging.info(f"Resuming from existing checkpoint: {summary_csv_path}")
        df_resume = pd.read_csv(summary_csv_path)
        # Reconstruct processed_combinations set to avoid re-running finished jobs
        for _, row in df_resume.iterrows():
            combo = (row["circuit_name"].split('_')[0], row["n_qubits"], row["depth"], row["backend_name"], row["repetition_index"])
            processed_combinations.add(combo)
        logging.info(f"Resumed {len(processed_combinations)} completed benchmarks.")

    combinations = list(product(circ_names, qubit_counts, depths, backend_names, range(repetitions)))
    total_jobs = len(combinations)
    logging.info(f"Starting sweep with {total_jobs} total benchmark combinations.")

    start_time = time.time()

    for i, (circ_name, n_qubits, depth, backend_name, rep) in enumerate(combinations):
        job_key = (circ_name, n_qubits, depth, backend_name, rep)
        if job_key in processed_combinations:
            logging.info(f"[{i+1}/{total_jobs}] SKIPPING already completed: {job_key}")
            continue

        logging.info(f"[{i+1}/{total_jobs}] RUNNING: {job_key}")

        try:
            seed = seed_base + rep
            generator_func = load_circuit_generator(circ_name)
            backend_instance = load_backend(backend_name)
            circuit = generator_func(n_qubits=n_qubits, depth=depth, seed=seed)

            result = run_single_benchmark(circuit=circuit, backend=backend_instance, output_dir=str(raw_json_dir))

            if result.success:
                logging.info(f"  -> SUCCESS | Time: {result.total_time_seconds:.2f}s | Mem: {result.peak_gpu_memory_mb:.0f}MB")
            else:
                logging.warning(f"  -> FAILED | Reason: {result.error_message}")

            result_dict = dataclasses.asdict(result)
            result_dict["repetition_index"] = rep
            all_results.append(result_dict)

        except Exception as e:
            logging.error(f"  -> CRITICAL FAILURE in sweep loop for {job_key}: {e}", exc_info=True)
            failure_result = {"schema_version": "0.1.0", "backend_name": backend_name, "circuit_name": f"{circ_name}_{n_qubits}q_d{depth}", "n_qubits": n_qubits, "depth": depth, "success": False, "error_message": f"Sweep runner critical error: {e}", "repetition_index": rep}
            all_results.append(failure_result)

        # --- Checkpointing (Rule 15) ---
        if (i + 1) % 10 == 0 or (i + 1) == total_jobs:
            logging.info(f"--- CHECKPOINTING: Saving summary CSV with {len(all_results)} new records ---")
            if os.path.exists(summary_csv_path):
                df_existing = pd.read_csv(summary_csv_path)
                df_new = results_to_dataframe(all_results)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=["circuit_name", "backend_name", "repetition_index"], keep="last")
                df_combined.to_csv(summary_csv_path, index=False)
            else:
                df_new = results_to_dataframe(all_results)
                df_new.to_csv(summary_csv_path, index=False)

            all_results = [] # Clear in-memory list after writing

    total_duration = time.time() - start_time
    logging.info(f"Sweep finished in {total_duration / 3600:.2f} hours.")
    logging.info(f"Final summary saved to: {summary_csv_path}")


if __name__ == "__main__":
    config_file = Path(os.getenv("SWEEP_CONFIG_PATH", DEFAULT_SWEEP_CONFIG))
    run_sweep(config_file)