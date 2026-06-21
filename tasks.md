# OpenQSim — Phase 0A Tasks

_Last updated: 2026-06-21_

## Context
Phase 0A = collect benchmark data across circuits × qubits × depths × backends.
Full sweep target: **1320 combos** (`benchmark/sweep_config_0a.yaml`).

**GPU is unusable** on Kaggle T4: `qiskit-aer-gpu==0.15.1` raises
`cudaErrorNoKernelImageForDevice` (wheel arch ≠ card). → run **CPU-only** for now.

**Aer 0.17.2 segfaults** locally on parametrized 2-qubit rotation gates
(qft = controlled-phase, qaoa = rzz); only ghz (h+cx) survives. The project pins
**Aer 0.15.1** which is stable — so generate data on Kaggle (0.15.1), NOT locally (0.17.2).
Do not trust the local `data/benchmark_outputs/` partial run (213 files on 0.17.2).

## What's actually done (downloads/ = old Kaggle artifacts)
- **Only `ghz` × `aer_mps`** — 178 files, qubits 5–24. Nothing else exists.
- For the small band that's `ghz×mps` at {8,10,12,15}q = **60 combos done**.
- The 135 `aer_statevector` entries in the old `completed_combinations.txt` are
  PHANTOMS (marked done, zero output) — ignore; new run uses a fresh config hash.

## Split across two machines (no qubit overlap)
- **Kaggle CPU** → small band {8,10,12,15}  (`sweep_config_small.yaml`, 480)
- **Colab**      → high band {18,20,22,24,26,28} (`sweep_config_colab.yaml`, 720)
- Together cover the full 0a grid EXCEPT **5q** (see gap below).

## DONE — Small band 8–15q, Kaggle CPU (`sweep_config_small.yaml`)
- [x] 480/480 verified — 4 circuits × {8,10,12,15} × 5 depths × 2 backends × 3 reps.
- [x] All success, statevector fidelity = 1.0000 exact, entropy populated. P100 box, CPU compute.
- [x] Consolidated into `data/phase0a/raw/` (new overlays old sparse ghz/mps).
- Note: `Downloads/pwnetwork` (T4, ghz+qft only, partial) is SUPERSEDED — do not merge.

## Now — High band, Colab (`sweep_config_colab.yaml`, 720 combos)
- [ ] Colab clones main + runs `scripts/run_sweep.py --config sweep_config_colab.yaml`.
- [ ] Device auto-selected by AerConfig (GPU if Colab kernels run, else CPU).
- [ ] Verify with `scripts/verify_sweep.py --config sweep_config_colab.yaml`.

## Gap
- [ ] **5q** is in neither band. Mop up later (cheapest band, CPU, seconds).
      Add `5` to small config OR a tiny one-off run. Don't change small mid-run.

## Later
- [ ] Merge small + colab + 5q into the Phase 0A dataset; check vs full 0a grid.
- [ ] sweep_config_medium.yaml removed — superseded by colab band.

## Notebook state (done this session)
- [x] qiskit==1.4.2 pin kept (qft.py needs `QFT`, removed in Qiskit 2.1).
- [x] Split into 7 ordered cells; `_KEYS` paste-block in cell 1.
- [x] Cell 3: `SWEEP_NAME` selector + `OPENQSIM_DEVICE=CPU`.
- [x] Cell 5: GPU canary (skipped when CPU pinned); on GPU failure pins CPU.

## Open
- [ ] NVIDIA/Groq advisor keys optional — off = `advisor_predicted_backend` null.
