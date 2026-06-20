"""Remove failed QAOA records from checkpoint + raw dir so they re-run.

Run this BEFORE resuming the Kaggle sweep when:
  - The QAOA parameter-binding bug caused all QAOA records to fail
  - The checkpoint already marks those combo keys as "completed"
  - You want the fixed code to re-simulate them

Usage (local or in Kaggle notebook):
    python scripts/fix_failed_qaoa.py \
        --raw-dir  /kaggle/working/benchmark_outputs/raw \
        --ckpt-dir /kaggle/working/checkpoints

What it does:
  1. Scans raw JSON files for QAOA records with success=False
  2. Deletes those JSON files
  3. Removes their combo_key from completed_combinations.txt
  4. Adjusts last_completed_index and counts in sweep_checkpoint.json
"""

import argparse
import json
from pathlib import Path


def fix_failed_qaoa(raw_dir: str, ckpt_dir: str, dry_run: bool = False) -> None:
    raw_path = Path(raw_dir)
    ckpt_path = Path(ckpt_dir)
    completed_file = ckpt_path / "completed_combinations.txt"
    checkpoint_file = ckpt_path / "sweep_checkpoint.json"

    if not raw_path.exists():
        print(f"ERR raw dir not found: {raw_path}")
        return

    # 1. Find failed QAOA JSONs
    failed_keys = []
    files_to_delete = []
    for f in sorted(raw_path.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        circuit_name = data.get("circuit_name", "")
        success = data.get("success", True)
        error_msg = data.get("error_message", "") or ""
        combo_key = data.get("combo_key", "") or f.stem

        if "qaoa" in circuit_name.lower() and not success:
            failed_keys.append(combo_key)
            files_to_delete.append(f)

    print(f"Found {len(failed_keys)} failed QAOA records to remove.")
    if not failed_keys:
        print("Nothing to do.")
        return

    for f in files_to_delete:
        print(f"  {'[DRY RUN] Would delete' if dry_run else 'Deleting'}: {f.name}")
        if not dry_run:
            f.unlink()

    # 2. Remove combo keys from completed_combinations.txt
    if completed_file.exists():
        lines = completed_file.read_text().splitlines()
        failed_set = set(failed_keys)
        kept = [ln for ln in lines if ln.strip() not in failed_set]
        removed = len(lines) - len(kept)
        print(f"Removing {removed} keys from {completed_file.name}")
        if not dry_run:
            completed_file.write_text("\n".join(kept) + ("\n" if kept else ""))
    else:
        print(f"WARN {completed_file} not found — skipping key removal")

    # 3. Adjust sweep_checkpoint.json counts
    if checkpoint_file.exists():
        ckpt = json.loads(checkpoint_file.read_text())
        old_completed = ckpt.get("completed_count", 0)
        old_errors = ckpt.get("error_count", 0)
        new_completed = max(0, old_completed - len(failed_keys))
        new_errors = max(0, old_errors - len(failed_keys))
        print(
            f"Adjusting checkpoint: completed {old_completed} -> {new_completed}, "
            f"errors {old_errors} -> {new_errors}"
        )
        if not dry_run:
            ckpt["completed_count"] = new_completed
            ckpt["error_count"] = new_errors
            checkpoint_file.write_text(json.dumps(ckpt, indent=2))
    else:
        print(f"WARN {checkpoint_file} not found — skipping count adjustment")

    if dry_run:
        print("\n[DRY RUN] No changes written. Re-run without --dry-run to apply.")
    else:
        print(
            f"\n[OK] Done. {len(failed_keys)} failed QAOA records removed.\n"
            f"     The sweep will re-run them with the fixed QAOA generator."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove failed QAOA records from checkpoint")
    parser.add_argument(
        "--raw-dir",
        default="/kaggle/working/benchmark_outputs/raw",
        help="Directory containing raw benchmark JSON files",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="/kaggle/working/checkpoints",
        help="Directory containing sweep_checkpoint.json and completed_combinations.txt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    args = parser.parse_args()
    fix_failed_qaoa(args.raw_dir, args.ckpt_dir, dry_run=args.dry_run)
