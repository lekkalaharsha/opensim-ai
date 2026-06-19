---

# OpenQSim-AI: Project Scope & Execution Directive

**Version:** 2.0  
**Date:** 2026-06-19  
**Status:** Phase 0 — Active (Milestone 0 in progress)

---

## Project Identity

**Mission:** Build an AI-assisted research platform for comparing and optimizing quantum simulation techniques.

**What We Build:** The intelligent orchestration layer on top of existing quantum simulators. Not another simulator.

**What We Do NOT Build:**
- Custom quantum simulators
- AI-based quantum state compression
- Neural quantum state frameworks (we benchmark them, not build them)

**Public Identity:** OpenQSim Benchmark Suite  
**Tagline:** *An open-source dataset and benchmarking framework for comparing quantum simulation backends.*

---

## Team Roles (Updated)

| Role | Owner | Responsibility |
|:---|:---|:---|
| **Builder** | Claude Code | All code implementation. Backend abstraction, benchmark runner, metrics collector, dataset pipeline, dashboard, Kaggle integration, scripts. |
| **Testers** | Codex, Antigravity | Validation tests, stress tests, edge case discovery, regression testing. |
| **Research** | DeepSeek | Literature review, ML model architecture, entanglement oracle design, differentiable memory manager research, paper drafting. |
| **Project Managers** | ChatGPT, User | Strategic direction, scope approval, milestone definition, deliverable review, roadmap decisions. |

---

## Success Criterion (Phase 0–2)

> **"Given a QASM circuit, OpenQSim-AI predicts which simulator backend will run it most efficiently, with average runtime regret < 1.2× the optimal choice."**

Not “we simulated 40 qubits.” Not “we built a new simulator.” The measurable goal is **backend selection accuracy with minimal regret**.

---

## Repository Structure (Complete)

```
openqsim-ai/
├── backend/
│   ├── __init__.py
│   ├── abstract.py              # QuantumSimulatorBackend ABC + SimulationResult
│   ├── environment.py           # Environment metadata collector
│   ├── aer_statevector.py       # Qiskit Aer statevector wrapper
│   ├── aer_mps.py               # Qiskit Aer MPS wrapper (planned)
│   ├── aer_tensor_net.py        # Qiskit Aer tensor network wrapper (future)
│   ├── config.py                # GPU/fusion configuration
│   └── llm_advisor.py           # NVIDIA NIM LLM backend advisor (NEW)
│
├── benchmark/
│   ├── __init__.py
│   ├── circuit_library/
│   │   ├── __init__.py
│   │   ├── ghz.py               # GHZ generator
│   │   ├── qft.py               # QFT generator (planned)
│   │   ├── random.py            # Random circuit generator (planned)
│   │   ├── qaoa.py              # QAOA ansatz generator (planned)
│   │   ├── variational.py       # Hardware-efficient ansatz (planned)
│   │   └── clifford.py          # Clifford-only circuit generator (planned)
│   ├── runner.py                # Single & batch sweep runner
│   ├── metrics.py               # Timing, GPU memory, fidelity collection
│   ├── entanglement.py          # Tiered entanglement computation
│   ├── circuit_fingerprint.py   # ML feature extraction
│   ├── schema.py                # JSON schema validation
│   ├── checkpoint.py            # Checkpoint utility
│   ├── sweep_config_0a.yaml     # Phase 0A sweep configuration
│   └── sweep_config_0b.yaml     # Phase 0B (future)
│
├── kaggle/                      # Kaggle integration module (NEW)
│   ├── __init__.py
│   ├── environment.py           # Kaggle environment validation
│   ├── checkpoint.py            # Checkpoint manager
│   ├── runner.py                # KaggleRunner (sweep + persistence)
│   ├── dataset_assembler.py     # Raw JSONs → dataset assembler
│   └── api_client.py            # Kaggle Dataset API client
│
├── scripts/                     # Automation scripts (NEW)
│   ├── setup.sh                 # Linux/Mac setup
│   ├── setup.bat                # Windows setup
│   ├── run_sweep.sh             # Linux/Mac sweep runner
│   ├── run_sweep.bat            # Windows sweep runner
│   ├── run_sweep.py             # Cross-platform sweep runner
│   ├── test_nim_advisor.sh      # NVIDIA NIM test (Linux/Mac)
│   └── test_nim_advisor.bat     # NVIDIA NIM test (Windows)
│
├── frontend/
│   ├── dashboard.py             # Streamlit dashboard
│   └── visualizations/          # Plots
│
├── research/                    # ML models & research
│   ├── backend_selector/
│   │   ├── rule_baseline.py
│   │   ├── xgboost_model.py
│   │   └── inference.py
│   ├── bond_dimension_oracle/
│   ├── entanglement_predictor/
│   └── memory_manager/
│
├── tests/
│   ├── test_integration.py      # Milestone 0 gate
│   ├── test_nim_advisor.py      # NVIDIA NIM advisor test
│   ├── test_backend_abstract.py
│   ├── test_runner.py
│   ├── test_metrics.py
│   ├── test_kaggle_*.py
│   └── ...
│
├── data/
│   ├── raw/                     # Raw JSON outputs (immutable)
│   ├── processed/               # Cleaned data
│   └── datasets/
│       └── openqsim_v0.1-small/ # Versioned dataset
│           ├── circuits.json
│           ├── results.csv
│           ├── manifest.json
│           └── DATASET_CARD.md
│
├── docs/
│   ├── OPENSIM_SCOPE.md         # This document
│   ├── ROADMAP.md               # Phase roadmap
│   └── dataset_card.md
│
├── .github/
│   └── workflows/
│       └── test.yml             # CI/CD pipeline
│
├── .claude/                     # Claude Code configs
├── .githooks/                   # Pre-commit, pre-push hooks
├── .env.example                 # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Core Components (Implemented & Planned)

### ✅ Phase 0 – Implemented (Milestone 0)
- `SimulationResult` dataclass with full metrics
- `QuantumSimulatorBackend` ABC
- Environment metadata collection (GPU, versions)
- Aer statevector backend (timing, GPU memory polling, exact fidelity)
- GHZ circuit generator
- Single benchmark runner (circuit → JSON)
- Milestone 0 integration test (15‑point gate)
- **NVIDIA NIM LLM advisor** (`backend/llm_advisor.py`)
- **Automation scripts** for local execution (bash/batch/Python)

### 🔲 Phase 1 – Dataset Generation (Month 1)
- Complete circuit library (QFT, random, QAOA, variational, Clifford)
- Aer MPS backend
- Metrics module, entanglement computation, circuit fingerprint
- Schema validation
- Kaggle integration module (environment check, checkpoint, runner, assembler, API client)
- Sweep execution on Kaggle Tesla T4
- Dataset `openqsim_v0.1-small` (~1,000 records)

### 🔲 Phase 2 – Backend Selector (Month 2)
- Rule‑based baseline
- XGBoost / LightGBM models
- Inference API
- Streamlit dashboard
- **NVIDIA NIM advisor integrated as feature in selector**

### 🔲 Phase 3‑4 – Research Extensions
- Bond dimension oracle
- Entanglement growth predictor
- Differentiable quantum memory manager

---

## Environment Variables & Secrets

All credentials are stored in `.env` (never committed). Template: `.env.example`

```bash
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
NVIDIA_API_KEY=nvapi-...
OPENQSIM_KAGGLE_DATASET=username/openqsim-benchmarks
```

---

## CI/CD Pipeline

GitHub Actions (`.github/workflows/test.yml`) performs:
- Lint (flake8, black, isort)
- Unit tests (pytest)
- Integration tests (Milestone 0 gate)

Pre‑commit hooks enforce formatting, import ordering, and JSON schema validation.

---

## Success Metrics

| Milestone | Goal | Metric |
|:----------|:-----|:-------|
| M0 | Infrastructure | 15‑point integration test pass |
| M1 | Dataset | 1,000 valid records with full metadata |
| M2 | Backend Selector | Average runtime regret < 1.2× |

---
