# benchmark/checkpoint.py
"""Checkpoint utility for long-running benchmark sweeps.

Provides a lightweight Checkpoint class that can save/load sweep progress
to disk, enabling recovery after interruption.  Designed to work both
standalone (local runs) and as a building block for the Kaggle-specific
CheckpointManager in kaggle/checkpoint.py.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Set


class Checkpoint:
    """Simple file-based checkpoint for benchmark sweeps.

    Tracks the last completed combination index and maintains a set of
    completed combination keys to avoid duplicate work on resume.

    Usage:
        ckpt = Checkpoint("sweep_checkpoint.json")
        ckpt.save(index=42, completed_keys={"ghz_5q_d1_aer_sv_rep0", ...})
        ...
        state = ckpt.load()
        resume_from = state["last_index"] + 1
    """

    def __init__(self, filepath: str):
        """Initialize checkpoint with a file path.

        Args:
            filepath: Path to the checkpoint JSON file.
        """
        self.filepath = Path(filepath)
        self._completed_keys: Set[str] = set()

    def save(self, index: int, completed_keys: Set[str]) -> None:
        """Save checkpoint state to disk.

        Args:
            index: The index of the last successfully completed combination.
            completed_keys: Set of combination keys that have been processed.
        """
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "last_index": index,
            "completed_keys": sorted(completed_keys),
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.filepath, "w") as f:
            json.dump(state, f, indent=2)
        self._completed_keys = completed_keys.copy()

    def load(self) -> Dict:
        """Load checkpoint state from disk.

        Returns:
            Dictionary with keys 'last_index' (int) and 'completed_keys' (list).
            If no checkpoint exists, returns default empty state.
        """
        if not self.filepath.exists():
            return {"last_index": -1, "completed_keys": [], "timestamp": ""}

        with open(self.filepath, "r") as f:
            state = json.load(f)

        self._completed_keys = set(state.get("completed_keys", []))
        return state

    def is_completed(self, combo_key: str) -> bool:
        """Check if a specific combination key has already been completed.

        Args:
            combo_key: Unique identifier for the combination
                (e.g., 'ghz_5q_d1_aer_statevector_rep0').

        Returns:
            True if the combination is in the completed set.
        """
        # Ensure loaded state is current
        if not self._completed_keys and self.filepath.exists():
            self.load()
        return combo_key in self._completed_keys

    def mark_completed(self, combo_key: str) -> None:
        """Add a combination key to the in-memory completed set.

        Note: This does NOT write to disk immediately; call save() to persist.

        Args:
            combo_key: The combination key to mark as completed.
        """
        self._completed_keys.add(combo_key)

    def clear(self) -> None:
        """Delete the checkpoint file and reset in-memory state."""
        if self.filepath.exists():
            self.filepath.unlink()
        self._completed_keys.clear()

    @property
    def completed_count(self) -> int:
        """Number of completed combinations tracked."""
        if not self._completed_keys and self.filepath.exists():
            self.load()
        return len(self._completed_keys)