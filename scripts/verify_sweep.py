#!/usr/bin/env python
"""Verify a finished sweep against its config: expected vs present combos.

Usage:
    # download the Kaggle dataset first, then point --raw at the unzipped dir:
    python scripts/verify_sweep.py --config benchmark/sweep_config_small.yaml --raw <dir>

Reports total expected/present, the breakdown by backend/circuit/qubit, and lists
any missing (circuit, qubits, depth, backend, rep) combos. Exit 1 if incomplete.

ponytail: counts files by name, no JSON parsing — a present file == a done combo.
Add schema/success checks here only if a run starts emitting bad rows.
"""
import argparse
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import yaml


def expected_keys(cfg: dict) -> set[str]:
    reps = range(cfg["repetitions"])
    combos = product(cfg["circuits"], cfg["qubits"], cfg["depths"], cfg["backends"], reps)
    return {f"{c}_{q}q_d{d}_{b}_rep{r}" for c, q, d, b, r in combos}


def present_keys(raw: Path) -> set[str]:
    # match the runner's filename stem, ignore any aggregate files without _rep
    return {f.stem for f in raw.rglob("*.json") if "_rep" in f.stem}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", required=True, help="dir with the sweep's JSON results")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    raw = Path(args.raw)
    if not raw.exists():
        print(f"ERROR: raw dir not found: {raw}")
        return 1

    exp = expected_keys(cfg)
    got = present_keys(raw)
    missing = exp - got
    extra = got - exp  # present but not in this config (other bands — fine)

    print(f"config:   {args.config}")
    print(f"expected: {len(exp)}")
    print(f"present:  {len(exp & got)} / {len(exp)}  ({len(missing)} missing)")
    if extra:
        print(f"(+{len(extra)} files outside this config's band - ignored)")

    done = exp & got
    # key = "{circuit}_{q}q_d{d}_aer_{method}_rep{r}"
    circ = Counter(k.split("_")[0] for k in done)
    qub = Counter(k.split("_")[1] for k in done)
    back = Counter("_".join(k.split("_")[3:5]) for k in done)
    print(f"  by circuit: {dict(sorted(circ.items()))}")
    print(f"  by qubits:  {dict(sorted(qub.items()))}")
    print(f"  by backend: {dict(sorted(back.items()))}")

    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for k in sorted(missing)[:40]:
            print(f"  {k}")
        if len(missing) > 40:
            print(f"  ... and {len(missing) - 40} more")
        return 1
    print("\n[OK] sweep complete — all expected combos present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
