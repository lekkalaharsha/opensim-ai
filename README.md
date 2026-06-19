
# OpenQSim Benchmark Suite

**An open-source dataset and benchmarking framework for comparing quantum simulation backends.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.0+](https://img.shields.io/badge/qiskit-1.0+-purple.svg)](https://qiskit.org/)

---

## What is OpenQSim?

OpenQSim is a **benchmarking and AI‑assisted backend selector** for quantum circuit simulation.

We don't build yet another simulator. Instead, we provide:

- A **unified benchmark runner** that executes circuits across multiple Qiskit Aer backends (statevector, MPS, tensor network).
- A **versioned, documented dataset** of runtime, memory, fidelity, and entanglement metrics.
- An **AI backend selector** that, given a QASM circuit, predicts the most efficient backend *before* execution.

> **Goal:** *Given a QASM circuit, OpenQSim predicts which simulator backend will run it most efficiently, with average runtime regret < 1.2× the optimal choice.*

---

## Project Status

| Phase | Milestone | Status |
|:------|:----------|:------|
| **0** | Infrastructure Validation | 🚧 In Progress |
| 1 | Dataset `v0.1-small` (~1,000 records) | ⏳ Planned |
| 2 | Baseline Backend Selector (XGBoost) | ⏳ Planned |
| 3 | Research: Bond Dimension Oracle | ⏳ Planned |
| 4 | Research: Differentiable Memory Manager | ⏳ Planned |

---

## Quick Start

```bash
git clone https://github.com/your-org/openqsim-ai.git
cd openqsim-ai
pip install -r requirements.txt
```

Run the Milestone 0 integration test:

```bash
python -m pytest tests/test_integration.py -v
```

Or directly:

```python
from benchmark.circuit_library.ghz import generate_ghz_circuit
from backend.aer_statevector import AerStatevectorBackend
from benchmark.runner import run_single_benchmark

result = run_single_benchmark(generate_ghz_circuit(5), AerStatevectorBackend())
print(f"Fidelity: {result.fidelity:.6f}, Time: {result.total_time_seconds:.4f}s")
```

---

## Architecture

```
openqsim-ai/
├── backend/            # Simulator wrappers (ABC, Aer statevector, MPS, NVIDIA NIM)
├── benchmark/          # Runner, metrics, circuit library, fingerprint, schema
├── kaggle/             # Kaggle integration (checkpoint, runner, assembler, API)
├── scripts/            # Automation scripts (setup, sweep, test advisor)
├── frontend/           # Streamlit dashboard
├── research/           # ML models (backend selector, oracle, memory manager)
├── tests/              # Unit & integration tests
├── data/               # Raw outputs, processed data, versioned datasets
└── docs/               # Full scope, roadmap, dataset card
```

---

## Kaggle Workflow (GPU Benchmarks)

All heavy benchmarks run on Kaggle's free Tesla T4 GPUs.

1. **Set API keys** – Copy `.env.example` to `.env` and add your Kaggle and NVIDIA keys.
2. **Run setup** – `bash scripts/setup.sh` (Linux/Mac) or `scripts\setup.bat` (Windows).
3. **Execute sweep** – `bash scripts/run_sweep.sh` or in a Kaggle notebook:

```python
from kaggle import KaggleRunner
runner = KaggleRunner(
    sweep_config_path="benchmark/sweep_config_0a.yaml",
    output_dir="/kaggle/working/benchmark_outputs",
    kaggle_dataset="your-username/openqsim-benchmarks",
)
runner.run()
```

---

## Team

| Role | Owner |
|:------|:------|
| Builder | Claude Code |
| Testers | Codex, Antigravity |
| Research | DeepSeek |
| Management | ChatGPT, User |

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

**OpenQSim** – *Making quantum simulation predictable.*
```

---

All repository configuration files are now complete. The project is fully scaffolded and ready for implementation.