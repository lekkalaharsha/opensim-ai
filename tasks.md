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

## Now — Small band 8–16q, Kaggle CPU-only  (`sweep_config_small.yaml`, 480 combos)
- [ ] Run notebook on Kaggle (CPU pinned via `OPENQSIM_DEVICE=CPU`, cell 3).
- [ ] ~420 new combos: `ghz×statevector` + `qft/random/qaoa × both backends`.
- [ ] Verify: 480 small-band JSON, both backends, all 4 circuits, schema-valid.

## Next session — Medium band 16–24q, Kaggle CPU-only (`sweep_config_medium.yaml`, 480)
- [ ] Switch `SWEEP_NAME` in cell 3 → `sweep_config_medium.yaml`.
- [ ] qubits {18,20,22,24} (no 16 in grid); statevector up to 24q fits on CPU.

## Later
- [ ] High band 26–28q + GPU decision: find a T4-compatible `qiskit-aer-gpu`
      build (sm_75), else cap statevector at 24q for Phase 0A.
- [ ] Full `sweep_config_0a.yaml` only once GPU works (else it's CPU data anyway).

## Notebook state (done this session)
- [x] qiskit==1.4.2 pin kept (qft.py needs `QFT`, removed in Qiskit 2.1).
- [x] Split into 7 ordered cells; `_KEYS` paste-block in cell 1.
- [x] Cell 3: `SWEEP_NAME` selector + `OPENQSIM_DEVICE=CPU`.
- [x] Cell 5: GPU canary (skipped when CPU pinned); on GPU failure pins CPU.

## Open
- [ ] NVIDIA/Groq advisor keys optional — off = `advisor_predicted_backend` null.
