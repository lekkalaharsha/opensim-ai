"""Post-sweep QAOA re-run script.

Run this AFTER the main Phase 0A sweep completes to:
  1. Delete all failed QAOA raw JSONs
  2. Remove their combo keys from the checkpoint
  3. Re-run only QAOA circuits (with the fixed parameter-binding generator)
  4. Re-assemble the final dataset

Usage (in Kaggle notebook cell or terminal):
    python scripts/rerun_qaoa.py \
        --raw-dir  /kaggle/working/benchmark_outputs/raw \
        --ckpt-dir /kaggle/working/checkpoints \
        --dataset-dir /kaggle/working/benchmark_outputs/datasets/openqsim_v0.1-small \
        --repo-dir /kaggle/working/OpenQSim
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir",     default="/kaggle/working/benchmark_outputs/raw")
    parser.add_argument("--ckpt-dir",    default="/kaggle/working/checkpoints")
    parser.add_argument("--dataset-dir", default="/kaggle/working/benchmark_outputs/datasets/openqsim_v0.1-small")
    parser.add_argument("--repo-dir",    default="/kaggle/working/OpenQSim")
    parser.add_argument("--kaggle-dataset", default="", help="Kaggle dataset id for final push (optional)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, args.repo_dir)

    # ── Step 1: Remove failed QAOA records from raw dir + checkpoint ────────
    print("=" * 60)
    print("Step 1: Cleaning up failed QAOA records")
    print("=" * 60)
    from scripts.fix_failed_qaoa import fix_failed_qaoa
    fix_failed_qaoa(args.raw_dir, args.ckpt_dir, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[DRY RUN] Stopping here. Re-run without --dry-run to continue.")
        return

    # ── Step 2: Re-run QAOA circuits with fixed generator ───────────────────
    print("\n" + "=" * 60)
    print("Step 2: Re-running QAOA circuits")
    print("=" * 60)

    sweep_config = Path(args.repo_dir) / "benchmark" / "sweep_config_qaoa_rerun.yaml"
    if not sweep_config.exists():
        print(f"ERR Config not found: {sweep_config}")
        sys.exit(1)

    from kaggle.runner import KaggleRunner

    runner = KaggleRunner(
        sweep_config_path=str(sweep_config),
        output_dir=str(Path(args.raw_dir).parent),
        checkpoint_interval=10,
        artifact_interval=50,
        kaggle_dataset=args.kaggle_dataset or None,
        use_advisor=False,  # skip LLM calls for the targeted re-run
    )
    result = runner.run()

    print(f"\nQAOA re-run done: {result.completed_count}/{result.total_combinations}")
    print(f"  OOM: {result.oom_count}  Errors: {result.error_count}")
    print(f"  Time: {result.duration_seconds/60:.1f} min")

    if result.error_count > 0:
        print(f"WARN {result.error_count} QAOA records still failed — check logs above")

    # ── Step 3: Re-assemble final dataset ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Re-assembling dataset")
    print("=" * 60)

    from kaggle.dataset_assembler import DatasetAssembler
    manifest = DatasetAssembler(
        raw_dir=args.raw_dir,
        dataset_dir=args.dataset_dir,
    ).assemble()

    print(f"\nFinal dataset:")
    print(f"  Records:   {manifest.records}")
    print(f"  Successes: {manifest.successful_runs}")
    print(f"  OOM:       {manifest.oom_failures}")
    print(f"  Backends:  {manifest.backends}")
    print(f"  Qubits:    {manifest.qubit_range}")
    print(f"  Written to: {args.dataset_dir}")

    # ── Step 4 (optional): Push to Kaggle Dataset ───────────────────────────
    if args.kaggle_dataset:
        print("\n" + "=" * 60)
        print("Step 4: Pushing to Kaggle Dataset")
        print("=" * 60)
        import json, shutil, subprocess
        dataset_dir = Path(args.raw_dir).parent
        (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
            "title": "OpenQSim Benchmark Outputs",
            "id": args.kaggle_dataset,
            "licenses": [{"name": "MIT"}],
        }))
        r = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(dataset_dir),
             "-m", f"Phase 0A final: {manifest.records} records (QAOA re-run complete)"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print(f"[OK] Pushed to {args.kaggle_dataset}")
        else:
            zip_path = shutil.make_archive(
                str(dataset_dir.parent / "openqsim_final"), "zip", str(dataset_dir)
            )
            print(f"WARN Push failed: {r.stderr[:200]}")
            print(f"[OK] Zip saved: {zip_path} — download from Output tab")

    print("\n=== QAOA re-run complete ===")


if __name__ == "__main__":
    main()
