#!/usr/bin/env python
"""Organize flat sweep results into qubits/depths/circuits/backends subfolders.

    python scripts/organize_data.py --raw data/phase0a/raw --dest data/phase0a/organized

Tree:  <dest>/<NN>q/d<NN>/<circuit>/<backend>/<file>.json   (qubit & depth zero-padded
to 2 digits so they sort naturally). Copies by default — pass --move to move.

Idempotent: re-run as more data lands; existing files are overwritten in place.
"""
import argparse
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

# {circuit}_{q}q_d{depth}_{aer_method}_rep{r}.json
PAT = re.compile(r"^(?P<c>[a-z]+)_(?P<q>\d+)q_d(?P<d>\d+)_(?P<b>aer_[a-z]+)_rep(?P<r>\d+)\.json$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    ap.add_argument("--no-clobber", action="store_true",
                    help="skip files whose target already exists (fill gaps only)")
    args = ap.parse_args()

    raw, dest = Path(args.raw), Path(args.dest)
    files = [f for f in raw.rglob("*.json") if PAT.match(f.name)]
    if not files:
        print(f"no result files matched under {raw}")
        return 1

    moved, skipped = 0, []
    for f in files:
        m = PAT.match(f.name)
        if not m:
            skipped.append(f.name)
            continue
        sub = dest / f"{int(m['q']):02d}q" / f"d{int(m['d']):02d}" / m["c"] / m["b"]
        sub.mkdir(parents=True, exist_ok=True)
        target = sub / f.name
        if args.no_clobber and target.exists():
            skipped.append(f.name)
            continue
        (shutil.move if args.move else shutil.copy2)(str(f), str(target))
        moved += 1

    parsed = [PAT.match(f.name) for f in files]
    verb = "moved" if args.move else "copied"
    qubits = sorted({"%02dq" % int(m["q"]) for m in parsed})
    print(f"{verb} {moved} files -> {dest}")
    print(f"  qubits:   {qubits}")
    print(f"  circuits: {dict(sorted(Counter(m['c'] for m in parsed).items()))}")
    print(f"  backends: {dict(sorted(Counter(m['b'] for m in parsed).items()))}")
    if skipped:
        print(f"  skipped (unparseable): {len(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
