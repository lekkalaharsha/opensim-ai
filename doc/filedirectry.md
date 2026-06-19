openqsim-ai/
├── backend/
│   ├── __init__.py
│   ├── abstract.py              # QuantumSimulatorBackend ABC + SimulationResult
│   ├── environment.py           # GPU / version metadata collection
│   ├── aer_statevector.py       # Qiskit Aer statevector wrapper
│   ├── aer_mps.py               # Qiskit Aer MPS wrapper
│   ├── aer_tensor_net.py        # (future) Qiskit Aer tensor network wrapper
│   ├── config.py                # GPU allocation, fusion thresholds
│   └── llm_advisor.py           # NVIDIA NIM LLM backend advisor (NEW)
│
├── benchmark/
│   ├── __init__.py
│   ├── circuit_library/
│   │   ├── __init__.py
│   │   ├── ghz.py               # GHZ generator
│   │   ├── qft.py               # QFT generator
│   │   ├── random.py            # Random circuit generator
│   │   ├── qaoa.py              # QAOA ansatz generator
│   │   ├── variational.py       # Hardware-efficient ansatz
│   │   └── clifford.py          # Clifford-only circuit generator
│   ├── runner.py                # Single & batch sweep runner
│   ├── metrics.py               # Timing, GPU memory, fidelity
│   ├── entanglement.py          # Tiered entanglement computation
│   ├── circuit_fingerprint.py   # ML feature extraction
│   ├── schema.py                # JSON schema validation
│   ├── checkpoint.py            # Kaggle checkpointing utility
│   ├── sweep_config_0a.yaml     # Phase 0A sweep parameters
│   └── sweep_config_0b.yaml     # Phase 0B (future)
│
├── kaggle/                      # Kaggle integration module (NEW)
│   ├── __init__.py
│   ├── environment.py           # Kaggle environment validation
│   ├── checkpoint.py            # Checkpoint manager
│   ├── runner.py                # KaggleRunner (sweep + persistence)
│   ├── dataset_assembler.py     # Assemble raw JSONs into dataset
│   └── api_client.py            # Kaggle Dataset API client
│
├── frontend/
│   ├── dashboard.py             # Streamlit dashboard
│   └── visualizations/
│       ├── time_vs_qubits.py
│       ├── memory_vs_qubits.py
│       └── backend_comparison.py
│
├── data/
│   ├── raw/                     # Raw JSON outputs (immutable)
│   ├── processed/               # Cleaned CSV/Parquet for ML
│   └── datasets/
│       └── openqsim_v0.1-small/ # Versioned dataset release
│           ├── circuits.json
│           ├── results.csv
│           ├── manifest.json
│           └── DATASET_CARD.md
│
├── research/
│   ├── backend_selector/
│   │   ├── rule_baseline.py     # Heuristic baseline
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   └── inference.py         # Unified inference API
│   ├── bond_dimension_oracle/
│   │   └── oracle.py
│   ├── entanglement_predictor/
│   │   └── predictor.py
│   └── memory_manager/
│       └── policy.py
│
├── tests/
│   ├── __init__.py
│   ├── test_backend_abstract.py
│   ├── test_environment.py
│   ├── test_aer_statevector.py
│   ├── test_aer_mps.py
│   ├── test_nim_advisor.py      # NVIDIA NIM advisor tests (NEW)
│   ├── test_runner.py
│   ├── test_metrics.py
│   ├── test_entanglement.py
│   ├── test_circuit_fingerprint.py
│   ├── test_schema.py
│   ├── test_checkpoint.py
│   ├── test_kaggle_environment.py
│   ├── test_kaggle_checkpoint.py
│   └── test_integration.py
│
├── docs/
│   ├── OPENSIM_SCOPE.md         # Full project scope & roadmap
│   ├── dataset_card.md
│   └── ROADMAP.md               # Phase-by-phase deployment plan
│
├── .claude/                     # Claude Code agent configs
│   ├── skills.md
│   ├── rules.md
│   ├── agents.md
│   ├── hooks.md
│   ├── config.yaml
│   └── checklists/
│       ├── milestone-0.md
│       └── milestone-1.md
│
├── .github/                     # CI/CD
│   └── workflows/
│       └── test.yml
│
├── .githooks/
│   ├── pre-commit
│   ├── pre-push
│   └── post-commit
│
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── LICENSE
└── README.md