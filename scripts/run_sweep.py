#!/usr/bin/env python3
"""Standalone sweep runner for OpenQSim.

Can be executed directly on any system with a GPU (or CPU fallback).
Does not require Jupyter/Kaggle web interface.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure openqsim is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kaggle import KaggleRunner
from kaggle.dataset_assembler import DatasetAssembler
from kaggle.api_client import KaggleAPIClient


def main():
    parser = argparse.ArgumentParser(
        description="OpenQSim Benchmark Sweep Runner"
    )
    parser.add_argument(
        "--config",
        default="benchmark/sweep_config_0a.yaml",
        help="Path to sweep configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmark_outputs",
        help="Directory for benchmark outputs",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N records",
    )
    parser.add_argument(
        "--artifact-interval",
        type=int,
        default=50,
        help="Push to Kaggle Dataset every N records",
    )
    parser.add_argument(
        "--kaggle-dataset",
        help="Kaggle Dataset identifier (e.g., username/openqsim-benchmarks)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to Kaggle Dataset (local run only)",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Only assemble existing raw data into dataset",
    )
    args = parser.parse_args()

    # Determine Kaggle dataset name
    kaggle_dataset = args.kaggle_dataset
    if not kaggle_dataset and not args.no_push:
        username = os.environ.get("KAGGLE_USERNAME")
        if username:
            kaggle_dataset = f"{username}/openqsim-benchmarks"

    # If assemble only, skip sweep
    if args.assemble_only:
        raw_dir = os.path.join(args.output_dir, "raw")
        dataset_dir = os.path.join(args.output_dir, "datasets", "openqsim_v0.1-small")
        assembler = DatasetAssembler(raw_dir=raw_dir, dataset_dir=dataset_dir)
        manifest = assembler.assemble()
        print(f"[OK] Dataset assembled: {manifest.records} records")
        return

    # Run sweep
    runner = KaggleRunner(
        sweep_config_path=args.config,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
        artifact_interval=args.artifact_interval,
        kaggle_dataset=kaggle_dataset if not args.no_push else None,
    )

    print("Starting benchmark sweep...")
    result = runner.run()

    print(f"\nSweep complete: {result.completed_count} records")
    print(f"  Successful: {result.completed_count - result.oom_count - result.error_count}")
    print(f"  OOM: {result.oom_count}")
    print(f"  Errors: {result.error_count}")

    # Assemble dataset
    raw_dir = os.path.join(args.output_dir, "raw")
    dataset_dir = os.path.join(args.output_dir, "datasets", "openqsim_v0.1-small")
    assembler = DatasetAssembler(raw_dir=raw_dir, dataset_dir=dataset_dir)
    manifest = assembler.assemble()
    print(f"Dataset assembled: {manifest.records} records")

    # Push to Kaggle if requested
    if kaggle_dataset and not args.no_push:
        client = KaggleAPIClient(dataset=kaggle_dataset)
        try:
            client.push_benchmark_outputs(
                args.output_dir,
                message=f"Sweep complete: {manifest.records} records",
            )
            print(f"[OK] Pushed to Kaggle Dataset: {kaggle_dataset}")
        except Exception as e:
            print(f"WARN Kaggle push failed: {e}")
            print(f"   Data is in {args.output_dir}")


if __name__ == "__main__":
    main()