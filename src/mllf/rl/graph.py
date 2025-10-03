"""Graph structure for RL environment.

Each undirected edge stores four bias coefficients: linear, quadratic, skew, end.
Nodes are indexed 0..N-1. Edges are stored in an upper-triangle dictionary keyed by (i,j).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List

import numpy as np


@dataclass
class EdgeCoeffs:
    linear: float = 0.0
    quadratic: float = 0.0
    skew: float = 0.0
    end: float = 0.0


class Graph:
    """Simple undirected graph with coefficients on edges.

    Edges are keyed by (i,j) with i < j.
    """

    def __init__(self, num_nodes: int):
        if num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        self.num_nodes = num_nodes
        self.edges: Dict[Tuple[int, int], EdgeCoeffs] = {}
        # initialize fully-connected graph with zero coefficients
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                self.edges[(i, j)] = EdgeCoeffs()

    def set_edge(self, i: int, j: int, coeffs: EdgeCoeffs | List[float] | Tuple[float, float, float, float]):
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in self.edges:
            raise KeyError(f"Edge ({a},{b}) not present for {self.num_nodes} nodes")
        if isinstance(coeffs, EdgeCoeffs):
            self.edges[(a, b)] = coeffs
        else:
            self.edges[(a, b)] = EdgeCoeffs(*coeffs)

    def get_edge(self, i: int, j: int) -> EdgeCoeffs:
        a, b = (i, j) if i < j else (j, i)
        return self.edges[(a, b)]

    def as_vector(self) -> np.ndarray:
        """Return a flat vector of all edge coefficients in a consistent ordering.

        Ordering: for i in 0..n-2, for j in i+1..n-1, append [linear, quadratic, skew, end]
        """
        out = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                e = self.edges[(i, j)]
                out.extend([e.linear, e.quadratic, e.skew, e.end])
        return np.array(out, dtype=float)

    def from_vector(self, vec: List[float] | np.ndarray):
        """Populate edges from a flat vector following the same ordering as as_vector()."""
        arr = np.asarray(vec, dtype=float)
        expected = (self.num_nodes * (self.num_nodes - 1) // 2) * 4
        if arr.size != expected:
            raise ValueError(f"Vector length {arr.size} does not match expected {expected} for {self.num_nodes} nodes")
        idx = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[(i, j)] = EdgeCoeffs(float(arr[idx]), float(arr[idx+1]), float(arr[idx+2]), float(arr[idx+3]))
                idx += 4
