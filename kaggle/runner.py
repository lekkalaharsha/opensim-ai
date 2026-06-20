"""KaggleRunner – the main entry point for benchmark sweeps inside Kaggle.

Orchestrates: environment validation, sweep execution with checkpointing,
and optional push to a persistent Kaggle Dataset.
"""

import time
import traceback
import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import yaml

from benchmark.runner import run_single_benchmark
from benchmark.circuit_library.ghz import generate_ghz_circuit
from benchmark.circuit_library.qft import generate_qft_circuit
from benchmark.circuit_library.random import generate_random_circuit
from benchmark.circuit_library.qaoa import generate_qaoa_circuit
from benchmark.circuit_library.variational import generate_variational_circuit
from benchmark.circuit_library.clifford import generate_clifford_circuit

from backend.aer_statevector import AerStatevectorBackend
from backend.aer_mps import AerMPSBackend  # will raise NotImplementedError until built

from kaggle.environment import validate_kaggle_environment
from kaggle.checkpoint import CheckpointManager
from kaggle.api_client import KaggleAPIClient


CIRCUIT_GENERATORS = {
    "ghz": generate_ghz_circuit,
    "qft": generate_qft_circuit,
    "random": generate_random_circuit,
    "qaoa": generate_qaoa_circuit,
    "variational": generate_variational_circuit,
    "clifford": generate_clifford_circuit,
}

BACKENDS = {
    "aer_statevector": AerStatevectorBackend,
    "aer_mps": AerMPSBackend,
}


@dataclass
class KaggleSweepResult:
    """Summary returned after a sweep completes."""
    total_combinations: int
    completed_count: int
    oom_count: int
    error_count: int
    output_dir: str
    duration_seconds: float
    dataset_pushed: bool


class KaggleRunner:
    """Run a full benchmark sweep on Kaggle with checkpointing and persistence."""

    def __init__(
        self,
        sweep_config_path: str,
        output_dir: str = "/kaggle/working/benchmark_outputs",
        checkpoint_interval: int = 10,
        artifact_interval: int = 50,
        kaggle_dataset: Optional[str] = None,
    ):
        self.sweep_config_path = sweep_config_path
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.artifact_interval = artifact_interval
        self.kaggle_dataset = kaggle_dataset
        self.ckpt = CheckpointManager(str(self.output_dir.parent / "checkpoints"))

    def _load_config(self) -> dict:
        with open(self.sweep_config_path) as f:
            return yaml.safe_load(f)

    def _generate_combinations(self, config: dict) -> list:
        combos = []
        for c in config["circuits"]:
            for q in config["qubits"]:
                for d in config["depths"]:
                    for b in config["backends"]:
                        for r in range(config.get("repetitions", 1)):
                            seed = config.get("seed_base", 42) + r
                            key = f"{c}_{q}q_d{d}_{b}_rep{r}"
                            combos.append({
                                "combo_key": key,
                                "circuit": c,
                                "qubits": q,
                                "depth": d,
                                "backend": b,
                                "seed": seed,
                            })
        return combos

    def _save_result(self, combo, result) -> None:
        path = self.raw_dir / f"{combo['combo_key']}.json"
        env = result.environment
        record = {
            "schema_version": result.schema_version,
            "combo_key": combo["combo_key"],
            "circuit_name": result.circuit_name,
            "n_qubits": result.n_qubits,
            "depth": result.depth,
            "backend_name": result.backend_name,
            "compilation_time_seconds": result.compilation_time_seconds,
            "execution_time_seconds": result.execution_time_seconds,
            "total_time_seconds": result.total_time_seconds,
            "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
            "fidelity": result.fidelity,
            "fidelity_method": result.fidelity_method,
            "entropy": result.entropy,
            "entropy_method": result.entropy_method,
            "success": result.success,
            "error_message": result.error_message,
            "environment": {
                "python_version": env.python_version if env else "",
                "qiskit_version": env.qiskit_version if env else "",
                "qiskit_aer_version": env.qiskit_aer_version if env else "",
                "gpu_name": env.gpu_name if env else "",
                "gpu_memory_mb": env.gpu_memory_mb if env else 0,
                "cuda_version": env.cuda_version if env else "",
                "host_platform": env.host_platform if env else "",
            },
            "circuit_fingerprint": result.circuit_fingerprint,
        }
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def run(self) -> KaggleSweepResult:
        # 1. Validate environment
        report = validate_kaggle_environment()
        if not report.is_valid:
            raise RuntimeError("Kaggle environment invalid. Fix issues before running.")
        print("✅ Environment OK")

        config = self._load_config()
        combos = self._generate_combinations(config)
        total = len(combos)

        # 2. Resume from checkpoint
        ckpt = self.ckpt.load()
        start_idx = ckpt.last_completed_index + 1
        completed = ckpt.completed_count
        oom = ckpt.oom_count
        errors = ckpt.error_count

        print(f"Total combinations: {total}, resuming from {start_idx}")

        # 3. Hash config for checkpoint integrity
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]

        t0 = time.time()
        for i in range(start_idx, total):
            combo = combos[i]
            if self.ckpt.is_completed(combo["combo_key"]):
                continue

            print(f"[{i+1}/{total}] {combo['combo_key']} ...", end=" ")
            try:
                gen = CIRCUIT_GENERATORS[combo["circuit"]]
                circuit = gen(combo["qubits"], combo["depth"], combo["seed"])
                backend = BACKENDS[combo["backend"]]()
                result = run_single_benchmark(circuit, backend, str(self.raw_dir))
                self._save_result(combo, result)
                self.ckpt.mark_completed(combo["combo_key"])
                completed += 1
                if result.success:
                    print(f"✅ {result.total_time_seconds:.2f}s")
                elif result.error_message and "memory" in result.error_message.lower():
                    oom += 1
                    print(f"⚠️ OOM")
                else:
                    errors += 1
                    print(f"❌ {result.error_message or 'Unknown error'}")
            except Exception as e:
                errors += 1
                print(f"❌ {e}")
                self.ckpt.mark_completed(combo["combo_key"])

            if (i + 1) % self.checkpoint_interval == 0:
                self.ckpt.save(i, completed, oom, errors, total, config_hash)

            if self.kaggle_dataset and (i + 1) % self.artifact_interval == 0:
                try:
                    KaggleAPIClient(self.kaggle_dataset).push_benchmark_outputs(
                        str(self.output_dir),
                        f"Sweep checkpoint – {completed} records"
                    )
                except Exception as e:
                    print(f"⚠️ Artifact push failed: {e}")

        # final checkpoint & push
        self.ckpt.save(total - 1, completed, oom, errors, total, config_hash)
        pushed = False
        if self.kaggle_dataset:
            try:
                KaggleAPIClient(self.kaggle_dataset).push_benchmark_outputs(
                    str(self.output_dir),
                    f"Sweep complete – {completed} records"
                )
                pushed = True
            except Exception as e:
                print(f"⚠️ Final push failed: {e}")

        duration = time.time() - t0
        print(f"\n=== Sweep finished: {completed} records in {duration/3600:.1f}h ===")
        return KaggleSweepResult(
            total_combinations=total,
            completed_count=completed,
            oom_count=oom,
            error_count=errors,
            output_dir=str(self.raw_dir),
            duration_seconds=duration,
            dataset_pushed=pushed,
        )