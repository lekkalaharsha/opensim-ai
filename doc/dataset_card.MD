# Dataset Card – OpenQSim v0.1‑small

## Dataset Summary
OpenQSim‑v0.1‑small is a benchmark dataset for quantum circuit simulation. It contains performance measurements (runtime, GPU memory, fidelity, entanglement) for ~1,000 circuits executed on two Qiskit Aer backends: statevector and matrix product state (MPS). The goal is to enable data‑driven backend selection.

## Usage
- **Training backend selector models**
- **Analyzing entanglement scaling**
- **Comparing simulation strategies**
- **Reproducing quantum simulation benchmarks**

## Dataset Structure

## openqsim_v0.1-small/
  ├── circuits.json # Metadata for each unique circuit
  ├── results.csv # One row per benchmark run
  ├── manifest.json # Dataset version & statistics
  └── DATASET_CARD.md # This file



### results.csv Columns
| Column | Description |
|--------|-------------|
| combo_key | Unique combination identifier (e.g., ghz_5q_d1_aer_sv_rep0) |
| circuit_name | Circuit family (ghz, qft, random, qaoa) |
| n_qubits | Number of qubits |
| depth | Circuit depth |
| backend_name | Simulator backend used |
| compilation_time_seconds | Qiskit transpilation time |
| execution_time_seconds | Backend run time |
| total_time_seconds | Sum of compilation + execution |
| peak_gpu_memory_mb | Peak GPU memory usage (MB) |
| fidelity | Exact state fidelity (if statevector available) |
| fidelity_method | "exact" or null |
| entropy | Maximum bipartite entanglement entropy |
| entropy_method | "exact" (≤16 qubits) or "approximate" (17‑25) |
| success | Boolean, false if OOM |
| error_message | Error description if failed |

### circuits.json Fields
- circuit_id, circuit_name, n_qubits, depth, fingerprint (gate counts, interaction graph metrics)

## Source Data
All circuits were generated with fixed seeds and executed on an NVIDIA Tesla T4 GPU (15 GB) using Qiskit Aer 0.14+ and Python 3.11. The sweep configuration is available in `benchmark/sweep_config_0a.yaml`.

## Licensing
MIT License – freely usable for research and commercial purposes.

## Citation
If you use this dataset, please cite:
*OpenQSim Benchmark Suite, 2026. openqsim-ai.*