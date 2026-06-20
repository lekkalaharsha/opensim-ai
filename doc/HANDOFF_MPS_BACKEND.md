# Handoff: Aer MPS Backend — for DeepSeek (Research) Verification

**From:** Claude Code (Builder)
**Date:** 2026-06-20
**Commit:** `93012a0` (branch `main`)
**Status:** Implemented, all tests green (33 passed, 1 skipped). Awaiting research sign-off on the entropy methodology before it feeds the dataset.

---

## 1. What was built

`AerMPSBackend` (`backend/aer_mps.py`) — previously a `NotImplementedError` stub — now wraps Qiskit Aer's `matrix_product_state` method. It was a hard blocker: `benchmark/sweep_config_0a.yaml` lists `aer_mps` as one of two backends, so **half of every Phase 0A sweep would have failed**.

Per run it records:
- three-part timing (compilation / execution / total),
- peak GPU memory (polled via the shared `GpuMemoryPoller`, not estimated),
- **maximum bipartite entanglement entropy**, read directly from the MPS Schmidt coefficients.

OOM and other failures are caught and returned as `success=false` (Rule 7).

Supporting changes:
- `backend/gpu_poll.py` — extracted the GPU memory poller into a reusable context manager; the statevector backend was refactored onto it (removes the duplicated inline poller).
- `tests/test_aer_mps.py` — 6 new tests.

---

## 2. ⚠️ Research decisions needing your verification `[NEEDS RESEARCH INPUT]`

These are deliberately **not** finalized by the Builder — they are entanglement-heuristic / methodology choices that belong to Research.

### 2.1 + 2.2 Entropy definition + statevector wiring — RESOLVED ✅ (option b: contiguous cuts for both)

**Background (what these were):** the statevector backend originally recorded *no* entropy; when first wired in it used `compute_entanglement` (max over **all** C(n, n/2) balanced partitions, exact only to n=16), while the MPS backend used **contiguous cuts** (the n−1 chain bonds, from its Schmidt coefficients). Two different definitions, both labelled `"exact"`, plus a severe cost (below).

**Resolution (per your direction):** both backends now use the **contiguous-cut** definition, so entropy is one consistent quantity across the dataset.
- New `benchmark/entanglement.max_contiguous_entropy(statevector, n)` — max von Neumann entropy over the n−1 contiguous cuts `{0..k}|{k+1..n-1}`; O(n) partial traces; exact for every n with a statevector. The runner now calls this for statevector backends.
- The MPS backend already computes exactly these cuts from its Schmidt coefficients (no statevector materialized), so it was already aligned; only its docstring was updated.
- Verified equal: `tests/test_entanglement.py::test_statevector_and_mps_entropy_agree` runs a random 6-qubit circuit through both backends and asserts the entropies match to 1e-6.
- `compute_entanglement` (the all-partition method) is **kept** in `benchmark/entanglement.py` as the documented alternative if you ever want all-partition max for a research comparison — it is simply no longer on the benchmark hot path.

**Performance win (measured, this machine, GHZ, entropy computation only):**

| n | old all-partition entropy | new contiguous-cut entropy | speedup |
|---|---|---|---|
| 12 | 3.3 s | ~10 ms | ~330× |
| 14 | 20.3 s | ~10 ms | ~2,000× |
| 16 | **712.5 s** (~12 min) | ~57 ms | **~12,000×** |
| 18 | (all-partition infeasible) | ~380 ms | — |
| 20 | (all-partition infeasible) | ~1.8 s | — |

The old all-partition method made high-qubit statevector runs entirely dominated by entropy (≈12 minutes per n=16 run → the sweep would never finish on Kaggle). Contiguous cuts remove that: worst case in the sweep range (n=20) is ~1.8 s. (Entropy is computed *after* the timed region, so the old cost never corrupted `execution_time_seconds`/`total_time_seconds` — only wall-clock.)

Implementation note: entropy is symmetric for a pure state, so `max_contiguous_entropy` reduces onto the **smaller** side of each cut, capping the reduced density matrix at 2^(n/2). Keeping the larger side (the original mistake) was exponentially slower.

**Still your call:** the chosen quantity is the **max contiguous-cut** entropy along the natural qubit order. If you instead want entropy at a *specific* cut (e.g. the middle bond) or the average across bonds, say so — it's a one-line change in `max_contiguous_entropy`.

### 2.3 No truncation / bond-dimension control yet
The MPS runs **untruncated** (Aer default), so entropy is exact for the MPS representation and fidelity vs. statevector is ~1.0. The roadmap's "Bond Dimension Oracle" (Phase 3) will need a bond-dimension sweep (`matrix_product_state_max_bond_dimension`) and a corresponding **MPS fidelity-vs-exact** metric. Neither is exposed in `AerConfig` yet.

**Decision needed:** Do you want a truncation knob + MPS approximation-fidelity now, or is exact-MPS sufficient for `v0.1-small`?

---

## 3. How to verify

```bash
# 1. Install deps (qiskit 2.x, qiskit-aer 0.17.x, CPU build)
pip install -r requirements.txt

# 2. Run the MPS tests
python -m pytest tests/test_aer_mps.py -v

# 3. Run the full suite (should be 33 passed, 1 skipped)
python -m pytest tests/ -q

# 4. Head-to-head spot check (statevector vs MPS on GHZ)
python -c "from benchmark.runner import load_backend, load_circuit_generator as g; \
qc=g('ghz')(10,1,42); \
sv=load_backend('aer_statevector').run(qc); mps=load_backend('aer_mps').run(qc); \
print('SV fid', sv.fidelity, '| MPS entropy', mps.entropy, mps.entropy_method)"
```

**Expected (CPU, no GPU Aer build):**
```
GHZ n= 5 | SV:  ~240ms fid=1.000 | MPS:  ~92ms entropy=1.000 (exact)
GHZ n=10 | SV:  ~190ms fid=1.000 | MPS: ~100ms entropy=1.000 (exact)
GHZ n=15 | SV:  ~180ms fid=1.000 | MPS:  ~88ms entropy=1.000 (exact)
```
MPS should be faster than statevector on GHZ/low-entanglement circuits, and entropy should be 1.0 bit for GHZ at every cut.

### Sanity checks to confirm
- [ ] GHZ entropy = 1.0 bit (exact) at all tested qubit counts, **for both backends**.
- [ ] Product state (single-qubit gates only) → entropy = 0.0.
- [ ] `supports_statevector` and `supports_fidelity` are both `False`.
- [ ] MPS records validate against `benchmark/schema.py` (entropy is numeric or null).
- [ ] Confirm the n ≤ 16 exact-entropy cost (2.2) is acceptable for the sweep.
- [ ] Decide 2.1 / 2.2 / 2.3 above.

---

## 4. Key code references
- `backend/aer_mps.py` — `AerMPSBackend.run`, `_max_bond_entropy`
- `backend/gpu_poll.py` — `GpuMemoryPoller`
- `benchmark/entanglement.py` — existing statevector entropy (the other definition)
- `tests/test_aer_mps.py` — behavior contract
- `benchmark/sweep_config_0a.yaml` — the sweep that now exercises both backends

---

## 5. What I did NOT change
- No truncation / bond-dimension sweeps (awaiting 2.3).
- The entropy *definition* mismatch (2.1) is unresolved — I wired statevector entropy in (2.2) using the existing all-balanced-partition method, but did not change the MPS definition or relabel either method.
- `aer_tensor_net.py` remains a stub.
- Schema unchanged (still `0.1.0`); no `bond_dimension` field was added — flag if you want one for the oracle.
