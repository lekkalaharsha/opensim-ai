"""Kaggle Dataset API client for pushing benchmark outputs to persistent storage."""

import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional


class KaggleAPIClient:
    """Push and pull benchmark data to/from Kaggle Datasets."""

    def __init__(self, dataset: str):
        self.dataset = dataset

    def dataset_exists(self) -> bool:
        """Check whether the Kaggle Dataset already exists and is accessible."""
        result = subprocess.run(
            ["kaggle", "datasets", "status", self.dataset],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0

    def push_benchmark_outputs(self, output_dir: str, message: Optional[str] = None) -> bool:
        path = Path(output_dir)
        metadata = path / "dataset-metadata.json"
        if not metadata.exists():
            meta = {
                "title": "OpenQSim Benchmark Outputs",
                "id": self.dataset,
                "licenses": [{"name": "MIT"}],
            }
            with open(metadata, "w") as f:
                json.dump(meta, f)

        msg = message or f"Benchmark run - {datetime.now().isoformat()}"
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(path), "-m", msg],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode == 0:
            print(f"[OK] Pushed to {self.dataset}")
            return True

        stderr = result.stderr or ""
        if "403" in stderr or "Forbidden" in stderr or "404" in stderr or "not found" in stderr.lower():
            print(
                f"\nWARN Kaggle push failed (403/404). The dataset '{self.dataset}' "
                f"may not exist yet.\n"
                f"  To create it: go to kaggle.com/datasets/add and create a new dataset "
                f"named '{self.dataset.split('/')[-1]}' under your account.\n"
                f"  Then re-run the sweep or call push_benchmark_outputs() again.\n"
            )
            zip_path = self._save_local_zip(path)
            if zip_path:
                print(
                    f"[OK] Data saved as zip: {zip_path}\n"
                    f"     Download it from the Kaggle file sidebar (Output tab)."
                )
            return False

        raise RuntimeError(f"Kaggle push failed: {stderr}")

    def _save_local_zip(self, output_dir: Path) -> Optional[str]:
        """Create a zip archive of the output directory for manual download."""
        try:
            zip_base = str(output_dir.parent / "benchmark_outputs_backup")
            zip_path = shutil.make_archive(zip_base, "zip", str(output_dir))
            return zip_path
        except Exception as e:
            print(f"WARN Could not create zip backup: {e}")
            return None
