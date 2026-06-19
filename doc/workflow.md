# OpenQSim: Project Goal & Vision

**The Classical Shadow of Quantum Systems**  
Simulating a quantum computer on classical hardware is exponentially expensive. A state of `n` qubits demands `2^n` complex numbers—a barrier that has stood since Feynman. Yet physically relevant states often live in a tiny corner of this vast Hilbert space, structured by entanglement. The key is **finding the right compressed representation** for the circuit at hand: statevector, matrix product states (MPS), tensor networks, or neural quantum states.

**The Problem**  
Today, researchers and engineers choose a simulator backend (Qiskit Aer statevector, MPS, tensor network, etc.) based on intuition or trial-and-error. No unified system exists to:

1. **Benchmark** different backends fairly and reproducibly.
2. **Capture** the performance landscape (time, memory, fidelity, entanglement) across circuit classes.
3. **Predict** the best backend before execution, saving GPU hours and frustration.

Without this, simulation work is fragmented, resource‑inefficient, and often hits the memory wall unexpectedly.

---

## OpenQSim’s Mission

> **Build an open‑source research platform that benchmarks quantum circuit simulators, publishes a labeled performance dataset, and trains an AI assistant to recommend the optimal backend for any given circuit—minimizing runtime regret.**

We are not building yet another simulator. We are building the **intelligence layer on top of existing simulators**.

---

## Our Approach: Three Layers, One Platform

| Layer | What It Does | Artifacts |
|:------|:-------------|:----------|
| **Benchmark Runner** | Executes circuits on multiple backends (statevector, MPS, etc.) under controlled conditions. Collects compile time, execution time, peak GPU memory, fidelity, and entanglement entropy. | `backend/`, `benchmark/`, `kaggle/` |
| **OpenQSim Dataset** | Versioned, documented, and reproducible. Each record contains circuit metadata (qubits, gates, interaction graph), performance metrics, and entanglement estimates. This is the **defensible asset**. | `data/datasets/openqsim_v0.1-small/` (~1,000 records) |
| **AI Backend Selector** | Given a QASM circuit, predicts the most efficient backend with a confidence score. Uses a hybrid of gradient‑boosted trees (XGBoost) and a reasoning LLM (NVIDIA NIM). | `research/backend_selector/` |

Together, they transform the simulation workflow from guesswork to data‑driven decision:

```
QASM Circuit → OpenQSim → "Use aer_mps. Predicted 12.3 s, 4.5 GB. Confidence 0.87."
```

---

## Why This Is Different

**1. Data‑First, Not Simulator‑First**  
Most quantum projects compete on simulation speed. OpenQSim competes on **data**—a publicly available, multi‑backend benchmark dataset that grows in value with every contribution.

**2. Regret, Not Accuracy**  
Our AI selector is trained to minimize *runtime regret*—how much slower your simulation runs compared to the optimal choice. A classifier that always says “statevector” would have high accuracy but terrible regret. We optimize the metric that actually matters.

**3. AI‑Assisted Reasoning with NVIDIA NIM**  
We leverage free LLM inference (Llama‑3.1, Nemotron) to *explain* backend recommendations and analyze entanglement structure. This isn’t just a black‑box model—it’s an interactive research assistant.

**4. Kaggle‑Native, Reproducible, Open**  
All heavy benchmarks run on free Tesla T4 GPUs in Kaggle notebooks. Checkpointing and automatic persistence to Kaggle Datasets ensure zero data loss and perfect reproducibility.

**5. Grounded in Real Quantum Simulation Challenges**  
Our work directly addresses the bottlenecks identified in the research literature:  
- The **memory wall** for statevector simulation.  
- The **bond‑dimension barrier** for MPS.  
- The **small‑tensor problem** that starves GPUs in tensor network contractions.  
- The need for a **unified comparison framework** across these methods.

---

## The Dataset: Our Defensible Asset

| Property | Phase 0A (v0.1‑small) | Phase 0B (v1.0) |
|:---------|:---------------------|:----------------|
| Circuits | GHZ, QFT, Random, QAOA | + Variational, Clifford, more |
| Qubits | 5–20 | 5–30+ |
| Backends | Aer statevector, MPS | + Tensor network, NQS |
| Records | ~1,000 | ~10,000+ |
| Metrics | Time, memory, fidelity, max entanglement entropy | + Layer‑wise entropy, bond dimension |

Each record is a JSON file with full environment metadata (GPU type, driver version, Qiskit version) and a unique circuit fingerprint—making it a scientific‑grade dataset.

---

## Roadmap & Future Research

| Phase | When | Milestone | Deliverable |
|:------|:-----|:----------|:------------|
| **M0** | Now | Infrastructure validation | One valid benchmark JSON |
| **M1** | Month 1 | First public dataset | `openqsim_v0.1-small` |
| **M2** | Month 2 | Baseline backend selector | XGBoost + rule baseline, regret <1.2× |
| **M3‑4** | Months 3‑6 | Bond dimension oracle | Predict required χ from circuit structure |
| **M5‑6** | Months 7‑12 | Entanglement growth predictor | Layer‑by‑layer entropy forecasting (GNNs/Transformers) |
| **M7** | Year 1+ | Differentiable Quantum Memory Manager | Learn qubit reordering policies to minimize memory (flagship paper) |

We **publish early** (dataset v0.1, then selector demo) to attract contributors and build the community before tackling the deep research extensions.

---

## Team & Execution

| Role | Agent / Human | Responsibility |
|:------|:-------------|:---------------|
| **Builder** | Claude Code | All code implementation |
| **Testers** | Codex, Antigravity | Validation, stress tests, edge case discovery |
| **Research & Management** | ChatGPT, User , DeepSeek | Strategic direction, milestone sign‑off, ML research, paper drafting |

This structure keeps implementation and research cleanly separated, preventing scope creep and ensuring execution discipline.

Here's the updated Team & Execution section with DeepSeek explicitly included:

---

## Team & Execution

| Role | Agent / Human | Responsibility |
|:------|:-------------|:---------------|
| **Builder** | Claude Code | All code implementation – backend wrappers, benchmark runner, Kaggle integration, dashboard |
| **Testers** | Codex, Antigravity | Validation (unit/integration tests), stress tests, edge case discovery, regression testing |
| **Research** | **DeepSeek** | Literature analysis, algorithm design, ML model architecture, entanglement oracle research, differentiable memory manager design, paper drafting |
| **Management & Strategy** | ChatGPT, User | Strategic direction, milestone sign-off, scope approval, project roadmap, final decision-making |

---

## The Bottom Line

OpenQSim transforms quantum circuit simulation from a guessing game into a **predictable, data‑driven process**. It’s an open, extensible framework that grows with the community—and the dataset alone is a valuable contribution to the field.

**What we’re building is not just a tool; it’s a new layer of the quantum software stack.**