# benchmark/entanglement.py
"""Tiered entanglement entropy computation.

For a given statevector, compute the maximum bipartite von Neumann entropy.
Uses exact evaluation of all balanced cuts for n_qubits <= 16, and an
approximate strategy (interaction graph bisection + random sampling) for
17 <= n_qubits <= 25.  Returns (entropy, method) where method is "exact",
"approximate", or None if computation is not possible.
"""

from typing import Optional, Tuple, List
import numpy as np
from qiskit.quantum_info import partial_trace, entropy
from qiskit.quantum_info import Statevector


def _balanced_partitions(n: int) -> List[Tuple[List[int], List[int]]]:
    """Generate all balanced bipartitions of n qubits.

    A balanced bipartition splits the qubits into two subsets of size floor(n/2)
    and ceil(n/2).  Yields tuples (subsystem_A, subsystem_B).

    Args:
        n: Total number of qubits.

    Returns:
        List of (A, B) where A and B are lists of qubit indices forming a partition.
    """
    from itertools import combinations

    qubits = list(range(n))
    half = n // 2
    partitions = []
    for A_indices in combinations(qubits, half):
        A = list(A_indices)
        B = [q for q in qubits if q not in A]
        partitions.append((A, B))
    return partitions


def _graph_bisection(graph) -> Tuple[List[int], List[int]]:
    """Heuristic bisection of an interaction graph.

    Uses a simple spectral clustering approach to find an approximate
    minimum‑cut balanced partition.

    Args:
        graph: networkx Graph representing qubit interactions.

    Returns:
        Two lists of qubit indices (A, B).
    """
    import networkx as nx
    import numpy as np

    if graph.number_of_nodes() < 2:
        return list(graph.nodes()), []
    try:
        fvec = nx.fiedler_vector(graph, method="lanczos")
    except Exception:
        fvec = np.random.randn(graph.number_of_nodes())
    median = np.median(fvec)
    A = [node for node, val in zip(graph.nodes(), fvec) if val <= median]
    B = [node for node, val in zip(graph.nodes(), fvec) if val > median]
    if not A or not B:
        A = list(graph.nodes())[:len(graph)//2]
        B = list(graph.nodes())[len(graph)//2:]
    return A, B


def _entropy_of_cut(statevector: Statevector, subsystem_A: List[int], n_qubits: int) -> float:
    """Compute von Neumann entropy of a given bipartition.

    In Qiskit 1.x, partial_trace(state, qargs) keeps the qubits listed in
    qargs and traces out the rest. So we pass subsystem_A directly.

    Args:
        statevector: The full statevector as a numpy array.
        subsystem_A: List of qubit indices forming subsystem A (the subsystem to keep).
        n_qubits: Total number of qubits.

    Returns:
        von Neumann entropy S(ρ_A).
    """
    if not subsystem_A:
        return 0.0

    rho_A = partial_trace(statevector, sorted(subsystem_A))
    return entropy(rho_A, base=2)


def compute_entanglement(
    statevector: Optional[Statevector],
    n_qubits: int,
    interaction_graph=None,
) -> Tuple[Optional[float], Optional[str]]:
    """Compute maximum bipartite entanglement entropy for a quantum state.

    Tiered approach:
        n <= 16  → evaluate all balanced partitions (exact).
        17 <= n <= 25 → use interaction graph bisection + 100 random cuts (approximate).
        n > 25  → return (None, None).

    Args:
        statevector: The full statevector array. If None, returns (None, None).
        n_qubits: Number of qubits.
        interaction_graph: networkx Graph for approximate bisection (optional).

    Returns:
        (max_entropy, method) where method is "exact", "approximate", or None.
    """
    if statevector is None:
        return None, None
    if n_qubits > 25:
        return None, None

    if n_qubits <= 16:
        # Exact: all balanced partitions
        partitions = _balanced_partitions(n_qubits)
        max_ent = 0.0
        for A, _ in partitions:
            ent = _entropy_of_cut(statevector, A, n_qubits)
            if ent > max_ent:
                max_ent = ent
        return max_ent, "exact"

    # Approximate regime (17–25)
    import networkx as nx
    import random

    if interaction_graph is None:
        g = nx.Graph()
        g.add_nodes_from(range(n_qubits))
        for i in range(n_qubits - 1):
            g.add_edge(i, i + 1)
    else:
        g = interaction_graph

    # Bisection based on interaction graph
    A_bisect, _ = _graph_bisection(g)
    max_ent = _entropy_of_cut(statevector, A_bisect, n_qubits)

    # Add 100 random balanced partitions
    qubits = list(range(n_qubits))
    half = n_qubits // 2
    for _ in range(100):
        A_rand = sorted(random.sample(qubits, half))
        ent = _entropy_of_cut(statevector, A_rand, n_qubits)
        if ent > max_ent:
            max_ent = ent

    return max_ent, "approximate"