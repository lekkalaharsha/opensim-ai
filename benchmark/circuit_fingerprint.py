# benchmark/circuit_fingerprint.py
"""Circuit fingerprint extraction for ML features.

Given a Qiskit QuantumCircuit, produces a dictionary of structural metrics
that can be used as input to a backend selector model.
"""

from typing import Dict, Any, Optional
import networkx as nx
from qiskit import QuantumCircuit


def _build_interaction_graph(circuit: QuantumCircuit) -> nx.Graph:
    """Create a weighted interaction graph from a quantum circuit.

    Nodes are qubits, edges represent two‑qubit gates.  Edge weight equals
    the number of two‑qubit gates acting on that pair.

    Args:
        circuit: Qiskit QuantumCircuit.

    Returns:
        networkx Graph with edge weights.
    """
    G = nx.Graph()
    G.add_nodes_from(range(circuit.num_qubits))
    for instruction in circuit.data:
        qubits = instruction.qubits
        if len(qubits) == 2:
            u = circuit.find_bit(qubits[0]).index
            v = circuit.find_bit(qubits[1]).index
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G


def extract_circuit_fingerprint(circuit: QuantumCircuit) -> Dict[str, Any]:
    """Extract structural features from a quantum circuit.

    Features include:
        - qubits, depth
        - gate counts (by type: h, cx, rz, rx, ry, measure, ...)
        - interaction graph metrics (diameter, average degree, max degree,
          connected components, algebraic connectivity)

    Args:
        circuit: Qiskit QuantumCircuit.

    Returns:
        Dictionary of fingerprint features.
    """
    # Basic counts
    ops = circuit.count_ops()
    gate_counts = {}
    for key, count in ops.items():
        gate_counts[key] = count

    # Build interaction graph
    G = _build_interaction_graph(circuit)

    # Graph metrics (handle small/unconnected graphs gracefully)
    n_nodes = G.number_of_nodes()
    if n_nodes > 1 and nx.is_connected(G):
        try:
            diameter = nx.diameter(G)
        except nx.NetworkXError:
            diameter = -1
    else:
        diameter = -1 if n_nodes > 1 else 0

    if n_nodes > 0:
        avg_degree = sum(dict(G.degree()).values()) / n_nodes
        max_degree = max(dict(G.degree()).values()) if n_nodes > 0 else 0
    else:
        avg_degree = 0.0
        max_degree = 0

    connected_components = nx.number_connected_components(G)

    # Algebraic connectivity (Fiedler eigenvalue), if connected and >1 node
    if n_nodes > 1 and connected_components == 1:
        try:
            alg_conn = nx.algebraic_connectivity(G, method="lanczos")
        except Exception:
            alg_conn = None
    else:
        alg_conn = 0.0 if n_nodes <= 1 else None

    return {
        "qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "gate_counts": gate_counts,
        "interaction_graph": {
            "diameter": diameter,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "connected_components": connected_components,
            "algebraic_connectivity": alg_conn,
        },
    }