# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install with extras (dev, ml, dashboard, kaggle)
pip install -e ".[dev,ml,dashboard,kaggle]"

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_integration.py -v

# Run a single test function
python -m pytest tests/test_integration.py::test_milestone_0_integration -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing

# Format and lint
black .
isort .
flake8 .
mypy .

# Run a benchmark sweep locally (CPU/GPU) via the CLI entry point
python scripts/run_sweep.py --config benchmark/sweep_config_0a.yaml --no-push

# Or via the wrapper scripts (load .env, push to Kaggle dataset)
bash scripts/run_sweep.sh benchmark/sweep_config_0a.yaml   # Linux/Mac
scripts\run_sweep.bat benchmark\sweep_config_0a.yaml       # Windows
```

## Architecture

### Data flow

A benchmark run follows this path:
1. A circuit generator in `benchmark/circuit_library/` creates a `QuantumCircuit`
2. `benchmark/runner.py` calls `backend.run(circuit)` and collects a `SimulationResult`
3. The runner extracts a circuit fingerprint via `benchmark/circuit_fingerprint.py`, attaches environment metadata from `backend/environment.py`, then writes a JSON file to `data/raw/`
4. `benchmark/schema.py` validates the JSON against the required schema
5. For sweeps, `kaggle/runner.py:KaggleRunner` (invoked via `scripts/run_sweep.py`) drives this loop over all combinations in a YAML config, with checkpointing and optional Kaggle dataset push

### Layer structure

- **`backend/`** — simulator wrappers only. `abstract.py` defines `QuantumSimulatorBackend` (ABC), `SimulationResult` (dataclass), and `EnvironmentMetadata`. All backends must implement this ABC. `backend/` never imports from `benchmark/`.
- **`benchmark/`** — runner, metrics, circuit library, fingerprint, schema. The runner is the only orchestrator; it imports backends but backends don't import runners.
- **`kaggle/`** — Kaggle-specific sweep runner, checkpointing, and dataset assembly. Writes to `/kaggle/working/`.
- **`research/`** — ML models (inference only; training is out of scope for the Builder role). `backend_selector/` holds XGBoost inference and a rule-based baseline. `entanglement_predictor/` and `memory_manager/` are stubs.
- **`frontend/`** — Streamlit dashboard visualizations.
- **`tests/`** — every module has a matching `test_*.py`. `test_integration.py` is the Milestone 0 gate (15 assertions).

### Key data structures (defined in `backend/abstract.py`)

- `SimulationResult` — single source of truth for all benchmark metrics. Always include `schema_version="0.1.0"` and a populated `environment` block. `fidelity` must be `null` (not a placeholder) when not computable.
- `EnvironmentMetadata` — hardware/software context captured at run time via `backend/environment.py`.

### Circuit fingerprint

`benchmark/circuit_fingerprint.py:extract_circuit_fingerprint()` returns gate counts plus interaction-graph metrics (diameter, algebraic connectivity, etc.) that serve as ML input features for the backend selector.

### Timing

Every result records three times: `compilation_time_seconds`, `execution_time_seconds`, and `total_time_seconds` (their sum). GPU memory is polled via `pynvml` in a background thread at 100 ms intervals, not estimated.

## Coding rules (from `.claude/rules.md`)

- Every backend implements `QuantumSimulatorBackend` ABC — no exceptions.
- `data/raw/` JSON files are immutable once written.
- Every JSON record must include `"schema_version": "0.1.0"` and a complete `environment` block.
- Timing is always three-part: compilation, execution, total.
- `fidelity` is `null` when not computable — no placeholder values.
- `MemoryError` is caught and recorded as `success=false`, never a crash.
- Every circuit generator accepts a `seed` parameter.
- `backend/` never imports from `benchmark/`.
- GPU memory is polled via `pynvml`, not estimated.
- Kaggle checkpointing: flush to disk every 10 records, upload artifact every 50.

## Builder scope

This project's Builder (Claude Code) implements infrastructure only. Flag any task that requires a research decision with `[NEEDS RESEARCH INPUT]` and propose a default — do not finalize decisions on model architectures, feature engineering, entanglement heuristics, or reward functions. Never build custom quantum simulators, quantum state compression, or ML training loops (inference wrappers only).

## Commit convention

`feat(module):`, `fix(module):`, `test(module):`, `docs:`, `chore:`
