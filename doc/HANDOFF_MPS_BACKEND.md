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

### 2.1 Entropy definition differs between backends
The two backends currently compute entanglement entropy with **different definitions**, both labelled `"exact"`:

| Backend | Source | Bipartitions considered |
|---|---|---|
| `aer_statevector` (`benchmark/entanglement.py`) | partial trace of statevector | **all balanced** partitions (n ≤ 16) |
| `aer_mps` (new) | MPS Schmidt coefficients | **contiguous cuts only** (the n−1 chain bonds) |

For GHZ both yield exactly 1.0 bit, so tests agree. But in general, **MPS contiguous-cut max ≤ all-partition max**. 

**Decision needed:** Should both use the same definition? Options:
- (a) Keep as-is, but relabel MPS method as `"mps_contiguous"` (not `"exact"`) so the dataset distinguishes them.
- (b) Standardize on contiguous cuts for both (cheaper, MPS-native).
- (c) Standardize on all-balanced for both (requires reconstructing a statevector from the MPS — defeats MPS for large n).

My default pick if you don't respond: **(a)** — it's honest and non-destructive.

### 2.2 Statevector backend records NO entropy today
`AerStatevectorBackend` computes fidelity but **never calls `compute_entanglement`**, so statevector dataset rows have `entropy = null` while MPS rows have a value. The `check_ghz_entropy` quality gate (`benchmark/quality.py`) only ever passes for MPS rows. This asymmetry affects the entanglement-predictor training target.

**Decision needed:** Should the statevector backend also populate entropy (via `benchmark/entanglement.py`)? I can wire it in once you confirm the definition from 2.1.

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
- [ ] GHZ entropy = 1.0 bit (exact) at all tested qubit counts.
- [ ] Product state (single-qubit gates only) → entropy = 0.0.
- [ ] `supports_statevector` and `supports_fidelity` are both `False`.
- [ ] MPS records validate against `benchmark/schema.py` (entropy is numeric or null).
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
- Did not touch the statevector entropy gap (awaiting 2.1/2.2).
- `aer_tensor_net.py` remains a stub.
- Schema unchanged (still `0.1.0`); no `bond_dimension` field was added — flag if you want one for the oracle.
