"""Aggregate raw benchmark JSONs into the OpenQSim dataset format."""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class DatasetManifest:
    dataset: str
    version: str
    records: int
    unique_circuits: int
    successful_runs: int
    oom_failures: int
    error_failures: int
    created_at: str
    schema_version: str
    git_commit: str
    backends: list
    circuits: list
    qubit_range: list
    depth_range: list


class DatasetAssembler:
    def __init__(self, raw_dir: str, dataset_dir: str, dataset_name: str = "openqsim_v0.1-small", version: str = "0.1.0", git_commit: str = "unknown"):
        self.raw_dir = Path(raw_dir)
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.version = version
        self.git_commit = git_commit

    def assemble(self) -> DatasetManifest:
        records = []
        circuits_map: Dict[str, dict] = {}

        for f in self.raw_dir.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
            records.append({
                "combo_key": data.get("combo_key", ""),
                "circuit_name": data["circuit_name"],
                "n_qubits": data["n_qubits"],
                "depth": data["depth"],
                "backend_name": data["backend_name"],
                "compilation_time_seconds": data["compilation_time_seconds"],
                "execution_time_seconds": data["execution_time_seconds"],
                "total_time_seconds": data["total_time_seconds"],
                "peak_gpu_memory_mb": data["peak_gpu_memory_mb"],
                "fidelity": data.get("fidelity"),
                "fidelity_method": data.get("fidelity_method"),
                "entropy": data.get("entropy"),
                "entropy_method": data.get("entropy_method"),
                "success": data["success"],
                "error_message": data.get("error_message"),
            })
            cid = f"{data['circuit_name']}_{data['n_qubits']}q_d{data['depth']}"
            if cid not in circuits_map:
                circuits_map[cid] = {
                    "circuit_id": cid,
                    "circuit_name": data["circuit_name"],
                    "n_qubits": data["n_qubits"],
                    "depth": data["depth"],
                    "fingerprint": data.get("circuit_fingerprint", {})
                }

        df = pd.DataFrame(records)
        df.to_csv(self.dataset_dir / "results.csv", index=False)

        with open(self.dataset_dir / "circuits.json", "w") as f:
            json.dump(list(circuits_map.values()), f, indent=2)

        success = df[df["success"] == True]
        oom = df[df["success"] == False]
        backends = sorted(df["backend_name"].unique().tolist())
        circs = sorted(df["circuit_name"].unique().tolist())
        qubits = sorted(df["n_qubits"].unique())
        depths = sorted(df["depth"].unique())

        manifest = DatasetManifest(
            dataset=self.dataset_name,
            version=self.version,
            records=len(df),
            unique_circuits=len(circuits_map),
            successful_runs=len(success),
            oom_failures=len(oom),
            error_failures=0,
            created_at=datetime.now().isoformat(),
            schema_version="0.1.0",
            git_commit=self.git_commit,
            backends=backends,
            circuits=circs,
            qubit_range=[min(qubits), max(qubits)] if qubits else [0,0],
            depth_range=[min(depths), max(depths)] if depths else [0,0],
        )

        with open(self.dataset_dir / "manifest.json", "w") as f:
            json.dump(asdict(manifest), f, indent=2)

        print(f"✅ Dataset assembled: {manifest.records} records")
        return manifest