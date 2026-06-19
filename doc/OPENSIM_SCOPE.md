# OpenQSim Benchmark Suite — Full Scope Document

**Version:** 2.0  
**Date:** 2026-06-19  
**Status:** Phase 0 — Active  

## Mission
Build an AI‑assisted research platform for comparing and optimizing quantum simulation techniques. We create the intelligent orchestration layer on top of existing simulators—not another simulator.

## Core Deliverables
1. **Benchmark Runner** – Executes QASM circuits on multiple Qiskit Aer backends (statevector, MPS, tensor network) and collects timing, memory, fidelity, and entanglement metrics.
2. **OpenQSim Dataset** – Versioned, documented, reproducible dataset of benchmark records. The defensible asset.
3. **AI Backend Selector** – Given a QASM circuit, predicts the optimal backend with average runtime regret < 1.2×.

## Repository Structure
openqsim-ai/
├── backend/           # Simulator wrappers (ABC, Aer statevector, MPS, NVIDIA NIM advisor)
├── benchmark/         # Runner, metrics, circuit library, fingerprint, schema, checkpoint
├── kaggle/            # Kaggle integration (environment check, checkpoint, runner, assembler, API client)
├── scripts/           # Automation scripts (setup, run sweep, test advisor)
├── frontend/          # Streamlit dashboard
├── research/          # ML models (backend selector, oracle, memory manager)
├── tests/             # Unit & integration tests
├── data/              # Raw outputs, processed data, versioned datasets
└── docs/              # Scope, roadmap, dataset card

## Team
| Role | Owner | Responsibility |
|------|-------|----------------|
| Builder | Claude Code | All code implementation |
| Testers | Codex, Antigravity | Validation, stress tests, edge cases |
| Research | DeepSeek | ML architecture, literature analysis, paper drafting |
| Management | ChatGPT, User | Strategy, scope approval, milestone sign‑off |

## Success Metric (Phase 2)
**"Given a QASM circuit, OpenQSim predicts which simulator backend will run it most efficiently, with average runtime regret < 1.2× the optimal choice."**

## Phases
- **Phase 0 (M0):** Infrastructure validation – one valid benchmark JSON.
- **Phase 1 (M1):** Dataset v0.1‑small (~1,000 records) generated on Kaggle.
- **Phase 2 (M2):** Baseline backend selector (XGBoost + rule baseline) with regret evaluation.
- **Phase 3‑4:** Research extensions – bond dimension oracle, entanglement predictor, differentiable quantum memory manager (paper target).

## Key Design Decisions
- Data‑first: benchmark dataset before any AI model.
- Regret, not accuracy, drives evaluation.
- Kaggle Tesla T4 for free GPU benchmarking.
- NVIDIA NIM free tier for LLM‑assisted backend reasoning.
- All raw data immutable, schema‑versioned, and environment‑tagged.