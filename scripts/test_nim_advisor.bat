@echo off
REM Test NVIDIA NIM advisor on Windows
REM Usage: scripts\test_nim_advisor.bat

if "%NVIDIA_API_KEY%"=="" (
    echo ERROR: NVIDIA_API_KEY not set. Run: set NVIDIA_API_KEY=nvapi-...
    exit /b 1
)

python -c "import os, sys; sys.path.insert(0, '.'); from backend.llm_advisor import NIMBackendAdvisor; advisor = NIMBackendAdvisor(api_key='%NVIDIA_API_KEY%'); fingerprint = {'qubits': 20, 'depth': 40, 'gate_counts': {'h': 20, 'cx': 120}, 'interaction_graph': {'diameter': 5}}; rec = advisor.recommend_backend(qasm='OPENQASM 2.0; include \"qelib1.inc\"; qreg q[20]; h q[0]; cx q[0], q[1];', fingerprint=fingerprint); print(f'Recommended: {rec.recommended_backend}'); print(f'Confidence: {rec.confidence:.2f}'); print('✅ NVIDIA NIM advisor works!')"