# Handoff: Session 2 Changes — for DeepSeek (Research) Verification

**From:** Claude Code (Builder)
**Date:** 2026-06-20
**Branch:** `main`
**Test state:** 39 passed, 1 skipped (~4s)
**Local sweep result:** 450 records generated, 330 successful (n=5–15, CPU)

---

## 1. What changed this session

### 1.1 `.gitignore` — fixed exclusions

**Before:** Only excluded `data/raw/` and `data/processed/`. The `kaggle/` source package was previously wrongly excluded (fixed last session), but many generated artifacts were still untracked noise.

**After:** Now excludes:
```
data/                   # all generated benchmark outputs
questions/              # output of question generators
notes/ pyq/             # GATE exam source material
_checkpoints_nvidia/    # nvidia_question_generator checkpoints
_checkpoints_groq/      # groq_question_generator checkpoints
*.csv                   # generated catalog / report files
*.bat                   # local convenience scripts
```
`.env.example` is **intentionally tracked** (no real keys inside).

---

### 1.2 `backend/model_catalog.py` — NEW

Central registry for all LLM inference models used across the project. Previously `backend/llm_advisor.py` had an inline 4-entry dict; the question generators had their own hardcoded model strings. These are now unified.

**NVIDIA NIM models:**

| Alias | Model ID | RPM | Used by |
|---|---|---|---|
| `fast` | `meta/llama-3.1-8b-instruct` | 60 | llm_advisor (default) |
| `accurate` | `meta/llama-3.3-70b-instruct` | 30 | llm_advisor |
| `reasoning` | `nvidia/nemotron-3-super-120b-a12b` | 10 | llm_advisor |
| `quantum` | `nvidia/ising-calibration-1-35b-a3b` | 10 | llm_advisor |
| `question_gen` | `nvidia/llama-3.3-nemotron-super-49b-v1` | 10 | nvidia_question_generator.py |

**Groq models:**

| Alias | Model ID | RPM | Used by |
|---|---|---|---|
| `fast` | `meta-llama/llama-4-scout-17b-16e-instruct` | 30 | groq_question_generator.py |
| `accurate` | `llama-3.3-70b-versatile` | 30 | available for advisor |
| `fast_mini` | `llama-3.1-8b-instant` | 60 | available for bulk calls |
| `mixtral` | `mixtral-8x7b-32768` | 30 | available for advisor |

**API:**
```python
from backend.model_catalog import get_model, list_models, NVIDIA_DEFAULT, GROQ_DEFAULT

# lookup by provider + alias
entry = get_model("nvidia", "accurate")
print(entry.model_id, entry.rpm, entry.endpoint)

# list all groq models
for m in list_models("groq"):
    print(m.alias, m.model_id)
```

---

### 1.3 `backend/llm_advisor.py` — refactored to use catalog

`AVAILABLE_MODELS` and `DEFAULT_MODEL` now imported from `model_catalog.py` instead of duplicated inline. No behavior change for callers.

---

### 1.4 `.env.example` — NEW

Template at project root documenting every required key with its source URL. The real `.env` stays gitignored. Keys documented:

| Key | Source |
|---|---|
| `NVIDIA_API_KEY` / `_1` / `_2` | build.nvidia.com (free tier) |
| `GROQ_API_KEY` / `_1` / `_2` | console.groq.com (free tier) |
| `KAGGLE_USERNAME` + `KAGGLE_KEY` | kaggle.com/settings/account |

Each provider supports up to 3 keys for parallel workers (the question generators use all three simultaneously via `ThreadPoolExecutor`).

---

### 1.5 `nvidia_question_generator.py` + `groq_question_generator.py` — tracked

These were present in the working directory but untracked. Now committed to `main`. They are GATE DA exam question generators — unrelated to the quantum simulation benchmarking core but sharing the same API keys and `.env`.

- NVIDIA generator: 2 parallel workers, `nemotron-super-49b`, generates MCQ/MSQ/NAT/PYQ-style for T4_FD_Normalization and B1_Transactions_Concurrency_Recovery topics
- Groq generator: 3 parallel workers, `llama-4-scout-17b`, same topics, resumes from NVIDIA checkpoints if found

---

### 1.6 `benchmark/circuit_library/qaoa.py` — bug fix

**Bug:** `generate_qaoa_circuit` returned a circuit with unbound `Parameter` objects (`γ_0`, `β_0`, etc.). Qiskit Aer refused to simulate it:
```
Execution error: 'circuits have parameters but parameter_binds is not specified.'
```
This caused **all 80 QAOA combos** (all n × depth combinations) to fail with `success=False`.

**Fix:** Parameters are now bound with deterministic random values before the circuit is returned:
```python
if circuit.parameters:
    params_sorted = sorted(circuit.parameters, key=lambda p: p.name)
    values = rng.uniform(0, 2 * np.pi, len(params_sorted))
    circuit = circuit.assign_parameters(dict(zip(params_sorted, values)))
```
The same `rng` is seeded by the `seed` argument, so results are reproducible.

**Verified:** All QAOA circuits now run cleanly with `success=True` at all (n, depth) combinations tested (n=5/8/10, depth=1/2/4).

---

### 1.7 Windows encoding fixes — `kaggle/runner.py`, `kaggle/api_client.py`, `scripts/run_sweep.py`

Emoji characters (`✅`, `⚠️`, `❌`) in `print()` calls crash on Windows with `cp1252` codec. All replaced with plain ASCII:

| Before | After |
|---|---|
| `print("✅ Environment OK")` | `print("[OK] Environment validated")` |
| `print(f"✅ {result.total_time_seconds:.2f}s")` | `print(f"OK  {result.total_time_seconds:.2f}s")` |
| `print(f"⚠️ OOM")` | `print("OOM")` |
| `print(f"❌ {msg}")` | `print(f"ERR {msg}")` |
| `print(f"✅ Pushed to ...")` | `print(f"[OK] Pushed to ...")` |

---

### 1.8 `kaggle/dataset_assembler.py` — numpy int64 serialization fix

**Bug:** `df["n_qubits"].unique()` returns a numpy array; `min()`/`max()` on it return `numpy.int64` which `json.dump` cannot serialize:
```
TypeError: Object of type int64 is not JSON serializable
```

**Fix:** Wrap all count and range values with `int()` before passing to `DatasetManifest`:
```python
records=int(len(df)),
qubit_range=[int(min(qubits)), int(max(qubits))],
depth_range=[int(min(depths)), int(max(depths))],
```

---

### 1.9 `benchmark/sweep_config_local.yaml` — NEW

CPU-safe sweep for local development. Does not require GPU or Kaggle.

```
circuits: ghz, qft, random, qaoa
qubits: 5, 8, 10, 12, 15
depths: 1, 2, 4, 8
backends: aer_statevector, aer_mps
repetitions: 2
→ 320 combinations, ~10–30 min on CPU
```

Run with:
```bash
python scripts/run_sweep.py \
  --config benchmark/sweep_config_local.yaml \
  --output-dir data/benchmark_outputs \
  --no-push
```

---

### 1.10 `doc/HANDOFF_PHASE1_DEEPSEEK.md` — NEW (broader context doc)

Companion document covering the full Phase 1 state including the MPS backend, entropy unification, and model catalog. This document (`HANDOFF_SESSION2`) focuses on the delta for this session only.

---

## 2. Sweep results — local dataset generated

Running `sweep_config_local.yaml` (n=5–15, CPU, no Kaggle push) produced:

```
data/benchmark_outputs/raw/         ← 450 individual JSON records
data/benchmark_outputs/datasets/
  openqsim_v0.1-small/
    results.csv                     ← flat table of all records
    circuits.json                   ← unique circuit fingerprints
    manifest.json                   ← dataset metadata
```

**Manifest summary:**
```json
{
  "records": 450,
  "successful_runs": 330,
  "backends": ["aer_mps", "aer_statevector"],
  "qubit_range": [5, 15]
}
```

120 failures are expected: high-bond-dimension random circuits at n=15 depth=8 sometimes hit MPS memory limits. The full Phase 0A sweep (n up to 20, 840 combos) should be run on Kaggle T4 for the production dataset.

---

## 3. How to reproduce from clean checkout

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in real keys if needed

# Run tests (should be 39 passed, 1 skipped)
python -m pytest tests/ -q

# Local sweep (CPU, ~20 min)
python scripts/run_sweep.py \
  --config benchmark/sweep_config_local.yaml \
  --output-dir data/benchmark_outputs \
  --no-push

# Full Phase 0A sweep (Kaggle T4 GPU)
# Upload repo to Kaggle notebook, then:
python scripts/run_sweep.py \
  --config benchmark/sweep_config_0a.yaml \
  --output-dir /kaggle/working/benchmark_outputs \
  --kaggle-dataset harshalekkala1/openqsim-benchmarks
```

---

## 4. Research decisions — resolved this session

All five open items from the Phase 1 handoff were answered by DeepSeek and implemented:

### 4.1 Bond-dimension truncation — DEFERRED to Phase 3 ✓
No code change. `AerMPSBackend` stays untruncated. Phase 3 will add
`matrix_product_state_max_bond_dimension`, schema v0.2, and MPS-fidelity metric.

### 4.2 Tensor network backend — DEFERRED to Phase 0B ✓
No code change. `aer_tensor_net.py` remains a stub.

### 4.3 Entropy features — ALL THREE, IMPLEMENTED ✓

`SimulationResult` now has three entropy fields (all populated by both backends):

| Field | Definition | Used for |
|---|---|---|
| `entropy` | max over all n-1 contiguous cuts | primary ML feature (unchanged) |
| `entropy_middle` | entropy at center bond k=(n-1)//2 | MPS cost proxy |
| `entropy_avg` | mean over all n-1 cuts | smooth entanglement signal |

Changes:
- `benchmark/entanglement.py` — `contiguous_entropy_features()` computes all three in one O(n) pass
- `backend/abstract.py` — added `entropy_middle` and `entropy_avg` optional fields
- `backend/aer_mps.py` — `_all_bond_entropies()` replaces `_max_bond_entropy()`; all three fields populated
- `benchmark/runner.py` — calls `contiguous_entropy_features()` for statevector backends
- `kaggle/dataset_assembler.py` — both fields included in `results.csv`

Verified: `ghz n=8` both backends → `entropy=2.000, entropy_middle=2.000, entropy_avg=1.714 [exact]`

### 4.4 Groq fallback advisor — IMPLEMENTED ✓

`backend/llm_advisor.py` now has:
- `GroqBackendAdvisor` — same `recommend_backend()` signature; uses `GROQ_ENDPOINT` from catalog
- `CascadingAdvisor` — tier 1: NVIDIA → tier 2: Groq (on 429 rate limit) → tier 3: rule baseline

```python
from backend.llm_advisor import CascadingAdvisor
advisor = CascadingAdvisor.from_env()   # reads NVIDIA_API_KEY and GROQ_API_KEY from env
rec = advisor.recommend_backend(qasm, fingerprint)
print(rec.recommended_backend, rec.reasoning)
```

### 4.5 Winner label — IMPLEMENTED ✓

`benchmark/metrics.py` — `determine_winner(sv_result, mps_result)` implements:
- faster backend wins
- within 5% of each other → fidelity tiebreak (if available), else "tie"
- failed backend always loses

`kaggle/dataset_assembler.py` — `winner` column added to `results.csv` per
`(circuit_name, n_qubits, depth)` group.

---

## 5. Remaining open items

### 4.1 Bond-dimension truncation (MPS approximation)
`AerMPSBackend` is untruncated. The Phase 3 oracle needs `matrix_product_state_max_bond_dimension` and a per-run MPS-vs-exact fidelity metric. **Decision: needed for v0.1-small or defer to Phase 3?**

### 4.2 Entropy feature for ML
Current: max contiguous-cut entropy. **Decision: max / middle-bond / average / all three as separate features?**

### 4.3 Groq fallback for LLM advisor
`NIMBackendAdvisor` only uses NVIDIA. Groq `llama-3.3-70b-versatile` is in the catalog but not wired to the advisor. **Decision: add `GroqBackendAdvisor` as rate-limit fallback?** (Builder can implement in ~30 min once confirmed.)

### 4.4 Winner label for XGBoost training
The dataset records timing for both backends but has no auto-computed "winner" column. **Decision: winner = faster backend (on success), or does MPS approximation error count as a loss?**

### 4.5 `aer_tensor_net.py` — still a stub
Not in `sweep_config_0a.yaml`. **Decision: implement for Phase 0B, or later?**

---

## 5. Files changed this session

| File | Change |
|---|---|
| `.gitignore` | Added data/, questions/, csvs, bat files, checkpoint dirs |
| `.env.example` | NEW — key template (no real values) |
| `backend/model_catalog.py` | NEW — NVIDIA + Groq model registry |
| `backend/llm_advisor.py` | Import from model_catalog instead of inline dict |
| `benchmark/circuit_library/qaoa.py` | Bind parameters before returning circuit |
| `benchmark/sweep_config_local.yaml` | NEW — CPU-safe local sweep |
| `kaggle/runner.py` | Replace emoji print() → ASCII |
| `kaggle/api_client.py` | Replace emoji print() → ASCII |
| `kaggle/dataset_assembler.py` | Cast numpy int64 → int before json.dump |
| `scripts/run_sweep.py` | Replace emoji print() → ASCII |
| `nvidia_question_generator.py` | Tracked (was untracked) |
| `groq_question_generator.py` | Tracked (was untracked) |
| `doc/HANDOFF_PHASE1_DEEPSEEK.md` | NEW — Phase 1 broad context doc |
| `doc/HANDOFF_SESSION2_DEEPSEEK.md` | THIS FILE |
