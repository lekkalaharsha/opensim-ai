"""Checkpoint manager for Kaggle sweeps.

Adds richer tracking (OOM counts, error counts, sweep config hash) on top
of the simple Checkpoint in benchmark/checkpoint.py.
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SweepCheckpoint:
    """State of a Kaggle benchmark sweep."""
    last_completed_index: int = -1
    completed_count: int = 0
    oom_count: int = 0
    error_count: int = 0
    total_combinations: int = 0
    sweep_config_hash: str = ""
    last_updated: str = ""


class CheckpointManager:
    """File-based checkpoint manager for long Kaggle sweeps.

    Usage in notebook:
        ckpt = CheckpointManager("/kaggle/working/checkpoints")
        ckpt.save(index, completed, oom, errors, total, hash)
        state = ckpt.load()
    """

    def __init__(self, checkpoint_dir: str = "/kaggle/working/checkpoints"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.dir / "sweep_checkpoint.json"
        self.completed_file = self.dir / "completed_combinations.txt"

    def save(
        self,
        index: int,
        completed_count: int,
        oom_count: int,
        error_count: int,
        total_combinations: int,
        sweep_config_hash: str,
    ) -> None:
        state = SweepCheckpoint(
            last_completed_index=index,
            completed_count=completed_count,
            oom_count=oom_count,
            error_count=error_count,
            total_combinations=total_combinations,
            sweep_config_hash=sweep_config_hash,
            last_updated=datetime.now().isoformat(),
        )
        with open(self.checkpoint_file, "w") as f:
            json.dump(asdict(state), f, indent=2)

    def load(self) -> SweepCheckpoint:
        if not self.checkpoint_file.exists():
            return SweepCheckpoint()
        with open(self.checkpoint_file) as f:
            data = json.load(f)
        return SweepCheckpoint(**data)

    def is_completed(self, combo_key: str) -> bool:
        if not self.completed_file.exists():
            return False
        with open(self.completed_file) as f:
            return combo_key in f.read().splitlines()

    def mark_completed(self, combo_key: str) -> None:
        with open(self.completed_file, "a") as f:
            f.write(combo_key + "\n")

    def clear(self) -> None:
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.completed_file.exists():
            self.completed_file.unlink()