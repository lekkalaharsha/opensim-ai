"""Unified inference API for the backend selector.

Loads the trained model and rule baseline, and provides a single
function `predict_backend` that takes a QASM circuit (or its fingerprint)
and returns a recommendation with confidence and predicted runtime.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.circuit_fingerprint import extract_circuit_fingerprint
from research.backend_selector.rule_baseline import rule_baseline_predict


class BackendSelector:
    """Combined backend selector using rule baseline and (optionally) XGBoost.

    If a trained model is provided, it overrides the rule baseline
    when the model's confidence is high enough.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the selector.

        Args:
            model_path: Path to a joblib file containing {'model': xgb_model, 'encoder': label_encoder}.
                        If None, uses only the rule baseline.
        """
        self.model = None
        self.encoder = None
        if model_path:
            import joblib
            data = joblib.load(model_path)
            self.model = data["model"]
            self.encoder = data["encoder"]

    def predict(
        self,
        qasm_or_circuit: Any,
        fingerprint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Recommend a backend for a given circuit.

        Args:
            qasm_or_circuit: A Qiskit QuantumCircuit or QASM string.
            fingerprint: Optional precomputed fingerprint. If None, it will be extracted.

        Returns:
            Dict with keys:
                recommended_backend (str): 'aer_statevector' or 'aer_mps'.
                confidence (float): 0-1 confidence score.
                predicted_runtime (float): estimated total time in seconds (if model available).
                method (str): 'rule_baseline' or 'xgboost'.
        """
        # Extract fingerprint if needed
        if fingerprint is None:
            if hasattr(qasm_or_circuit, "num_qubits"):  # QuantumCircuit
                fingerprint = extract_circuit_fingerprint(qasm_or_circuit)
            else:
                raise ValueError("A QuantumCircuit or a fingerprint dict is required.")

        n_qubits = fingerprint.get("qubits", 0)
        depth = fingerprint.get("depth", 0)

        # Rule baseline always available
        rule_pred = rule_baseline_predict(n_qubits)

        # If we have a trained model, use it
        if self.model is not None and self.encoder is not None:
            # Build feature vector (must match training)
            features = np.array([[n_qubits, depth]])  # basic features; extend if needed
            proba = self.model.predict_proba(features)[0]
            best_idx = np.argmax(proba)
            confidence = proba[best_idx]
            ml_pred = self.encoder.inverse_transform([best_idx])[0]

            # Only use ML prediction if confidence > 0.6, else fallback to rule
            if confidence > 0.6:
                return {
                    "recommended_backend": ml_pred,
                    "confidence": float(confidence),
                    "predicted_runtime": None,  # XGBoost doesn't directly predict time
                    "method": "xgboost",
                }

        # Default to rule baseline
        return {
            "recommended_backend": rule_pred,
            "confidence": 0.8,  # heuristic confidence
            "predicted_runtime": None,
            "method": "rule_baseline",
        }