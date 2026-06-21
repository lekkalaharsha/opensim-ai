#!/usr/bin/env python
"""Assemble data/raw/*.json into the results.csv + circuits.json dataset
that research/backend_selector/train.py expects.

    python -m research.backend_selector.assemble_dataset \
        --raw data/phase0a/organized --out data/benchmark_outputs/datasets/openqsim_v0.1

One row per run in results.csv (circuit_name, n_qubits, depth, backend_name,
success, total_time_seconds, fidelity). One fingerprint per circuit
(circuit_name, n_qubits, depth) in circuits.json, taken from whichever run
recorded it.

winner = backend with min total_time_seconds among {aer_statevector, aer_mps},
both successful, fidelity-guarded: statevector's fidelity must be ~1.0 if
present (an incorrect-but-fast statevector run is not a winner). 'undecidable'
if fewer than 2 candidate backends succeeded.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ORACLE = ["aer_statevector", "aer_mps"]
FIDELITY_TOL = 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir to walk for *.json results (recursive)")
    ap.add_argument("--out", required=True, help="output dataset dir (results.csv + circuits.json)")
    ap.add_argument("--max-qubits", type=int, default=None, help="drop circuits above this qubit count")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    files = [f for f in raw.rglob("*.json") if "_rep" in f.stem]
    if args.max_qubits is not None:
        files = [f for f in files if json.loads(f.read_text())["n_qubits"] <= args.max_qubits]
    if not files:
        print(f"no result files found under {raw}")
        return 1

    rows = []
    fingerprints = {}  # key -> fingerprint dict
    for f in files:
        rec = json.loads(f.read_text())
        cname, n, d = rec["circuit_name"], rec["n_qubits"], rec["depth"]
        rows.append({
            "circuit_name": cname,
            "n_qubits": n,
            "depth": d,
            "backend_name": rec["backend_name"],
            "success": rec["success"],
            "total_time_seconds": rec["total_time_seconds"],
            "fidelity": rec.get("fidelity"),
        })
        key = f"{cname}_{n}q_d{d}"
        fp = rec.get("circuit_fingerprint")
        if fp and key not in fingerprints:
            fingerprints[key] = {"circuit_name": cname, "n_qubits": n, "depth": d, "fingerprint": fp}

    df = pd.DataFrame(rows)

    winners = []
    for (cname, n, d), g in df.groupby(["circuit_name", "n_qubits", "depth"]):
        cand = g[g["backend_name"].isin(ORACLE) & g["success"]]
        ok = cand[cand["fidelity"].isna() | (cand["fidelity"] >= 1.0 - FIDELITY_TOL)]
        winner = ok.loc[ok["total_time_seconds"].idxmin(), "backend_name"] if len(ok) >= 2 else "undecidable"
        winners.append({"circuit_name": cname, "n_qubits": n, "depth": d, "winner": winner})

    df = df.merge(pd.DataFrame(winners), on=["circuit_name", "n_qubits", "depth"], how="left")

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    (out / "circuits.json").write_text(json.dumps(list(fingerprints.values())))

    decidable = (df.drop_duplicates(["circuit_name", "n_qubits", "depth"])["winner"] != "undecidable").sum()
    print(f"rows: {len(df)}  circuits: {df.groupby(['circuit_name','n_qubits','depth']).ngroups}  "
          f"decidable: {decidable}  fingerprints: {len(fingerprints)}")
    print(f"-> {out}/results.csv, {out}/circuits.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
