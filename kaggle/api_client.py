"""Kaggle Dataset API client for pushing benchmark outputs to persistent storage."""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional


class KaggleAPIClient:
    """Push and pull benchmark data to/from Kaggle Datasets."""

    def __init__(self, dataset: str):
        self.dataset = dataset

    def push_benchmark_outputs(self, output_dir: str, message: Optional[str] = None) -> bool:
        path = Path(output_dir)
        metadata = path / "dataset-metadata.json"
        if not metadata.exists():
            meta = {"title": "OpenQSim Benchmark Outputs", "id": self.dataset, "licenses": [{"name": "MIT"}]}
            with open(metadata, "w") as f:
                json.dump(meta, f)

        msg = message or f"Benchmark run – {datetime.now().isoformat()}"
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(path), "-m", msg],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle push failed: {result.stderr}")
        print(f"✅ Pushed to {self.dataset}")
        return True