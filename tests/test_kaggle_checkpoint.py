# tests/test_kaggle_checkpoint.py
"""Tests for Kaggle checkpoint manager."""

import tempfile
from kaggle.checkpoint import CheckpointManager


def test_save_and_load():
    with tempfile.TemporaryDirectory() as d:
        ckpt = CheckpointManager(d)
        ckpt.save(42, 43, 2, 1, 100, "abc123")
        state = ckpt.load()
        assert state.last_completed_index == 42
        assert state.completed_count == 43
        assert state.oom_count == 2
        assert state.error_count == 1
        assert state.total_combinations == 100


def test_is_completed():
    with tempfile.TemporaryDirectory() as d:
        ckpt = CheckpointManager(d)
        assert not ckpt.is_completed("test_combo")
        ckpt.mark_completed("test_combo")
        assert ckpt.is_completed("test_combo")


def test_clear():
    with tempfile.TemporaryDirectory() as d:
        ckpt = CheckpointManager(d)
        ckpt.save(1, 2, 0, 0, 5, "xyz")
        ckpt.clear()
        state = ckpt.load()
        assert state.last_completed_index == -1