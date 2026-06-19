# OpenQSim Development Roadmap

**Version:** 2.0  
**Date:** 2026-06-19  

## Phase 0 – Infrastructure Validation (Current)
**Milestone 0**  
- [x] Backend ABC & SimulationResult dataclass  
- [x] Environment metadata collector  
- [x] Aer statevector wrapper  
- [x] GHZ circuit generator  
- [x] Single benchmark runner  
- [ ] MPS backend stub  
- [ ] NVIDIA NIM advisor integration test  
- **Gate:** 15‑point integration test passes → one valid JSON file  

## Phase 1 – Dataset Generation (Month 1)
**Milestone 1**  
- Complete all six circuit generators (GHZ, QFT, random, QAOA, variational, Clifford)  
- Implement MPS backend  
- Metrics, entanglement, fingerprint, schema modules  
- Kaggle integration (environment check, checkpoint manager, KaggleRunner, dataset assembler, API client)  
- Run Phase 0A sweep on Kaggle (4 circuits × 7 qubits × 5 depths × 2 backends × 3 reps)  
- Assemble `openqsim_v0.1-small` (~1,000 records)  
- Push dataset to Kaggle Dataset and GitHub  

## Phase 2 – AI Backend Selector (Month 2)
**Milestone 2**  
- Rule‑based baseline  
- Train XGBoost / LightGBM on circuit fingerprints  
- Evaluate with runtime regret (target < 1.2×)  
- Build inference API (`predict_backend(qasm) → recommendation`)  
- Streamlit dashboard with backend selector tab  
- Integrate NVIDIA NIM advisor for explanation  

## Phase 3 – Research Extensions (Months 3‑6)
- **Bond Dimension Oracle:** Predict required MPS bond dimension from circuit structure (MAE < 20%)  
- **Entanglement Growth Predictor:** Layer‑by‑layer entropy forecasting using GNNs/Transformers (R² > 0.9)  

## Phase 4 – Flagship Research (Months 7‑12)
- **Differentiable Quantum Memory Manager:** RL‑based qubit reordering policy to minimize peak memory  
- Target venue: NeurIPS/ICML workshop paper  

## Release Plan
- v0.2.0 – First public dataset + benchmark framework  
- v0.3.0 – Backend selector demo  
- v1.0.0 – Full research platform with all AI features