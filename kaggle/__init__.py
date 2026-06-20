"""Kaggle integration layer for OpenQSim Benchmark Suite.

This module provides everything needed to run benchmark sweeps inside a
Kaggle notebook with checkpointing and persistent dataset storage.
"""

from kaggle.runner import KaggleRunner
from kaggle.checkpoint import CheckpointManager
from kaggle.environment import validate_kaggle_environment
from kaggle.dataset_assembler import DatasetAssembler
from kaggle.api_client import KaggleAPIClient

__all__ = [
    "KaggleRunner",
    "CheckpointManager",
    "validate_kaggle_environment",
    "DatasetAssembler",
    "KaggleAPIClient",
]