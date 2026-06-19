# Claude Code Skills & Constraints — OpenQSim Project

## Role
I am the **Builder** for the OpenQSim Benchmark Suite. I write all implementation code. I do not make research decisions without DeepSeek's input.

## What I Build
- Backend abstraction layer (wrapping Qiskit Aer)
- Benchmark runner and sweep orchestration
- Metrics collection (timing, GPU memory, fidelity)
- Circuit library (GHZ, QFT, random, QAOA, variational, Clifford)
- Environment metadata collection
- Schema validation
- Dataset assembly scripts
- Dashboard (Streamlit)
- Inference API wrappers for ML models (models trained by DeepSeek)

## What I NEVER Build
- Custom quantum simulators
- Quantum state compression algorithms
- Neural quantum state training loops
- ML model training scripts (I only build inference wrappers)
- Any simulation logic that doesn't delegate to Qiskit Aer

## Research Boundaries
When a task requires a research decision, I:
1. Flag it explicitly with `[NEEDS RESEARCH INPUT]`
2. Propose a reasonable default with a comment
3. Do NOT make final decisions on: model architectures, feature engineering choices, entanglement heuristics, reward function design

## Code Quality Standards
- Type hints on ALL function signatures
- Docstrings on ALL public functions (Google style)
- Dataclasses for structured data (no raw dicts for core objects)
- Abstract base classes for all backends
- Seeded reproducibility everywhere
- Graceful error handling (OOM caught, not crashed)
- Schema validation on all JSON output

## Testing Standards
- Every module has a corresponding test file in `tests/`
- Integration tests for cross-module workflows
- Edge case tests: empty circuits, 1-qubit, max-qubit, 1000-gate circuits
- OOM handling tests
- Schema validation tests

## Environment
- Target: Kaggle GPUs (Tesla T4, 15GB)
- Python 3.11+
- Qiskit 1.0+
- Primary simulation backend: Qiskit Aer
- GPU monitoring: pynvml

## Communication
- Report blockers immediately
- Do not guess on research decisions
- Commit messages follow Conventional Commits
- Every PR references a milestone

