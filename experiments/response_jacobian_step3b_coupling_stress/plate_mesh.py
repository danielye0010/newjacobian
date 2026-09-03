"""Fixed-connectivity structured thin-plate mesh for Step 3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy import sparse
except Exception:  # pragma: no cover
    sparse = None


@dataclass(frozen=True)
class PlateMesh:
    Lx: float
    Ly: float
    nx: int
    ny: int
    nodes: np.ndarray
    edges: np.ndarray
    boundary_mask: np.ndarray
    weights_node: np.ndarray
    weights_dof: np.ndarray
    laplacian: object
    dx: float
    dy: float

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_dofs(self) -> int:
        return int(self.nodes.size)

    @property
    def total_area(self) -> float:
        return float(np.sum(self.weights_node))

    def flatten(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float).reshape(-1)

    def unflatten(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v, dtype=float).reshape(self.num_nodes, 3)

    def diagnostics(self) -> Dict[str, object]:
        return {
            "Lx": self.Lx,
            "Ly": self.Ly,
            "nx": self.nx,
            "ny": self.ny,
            "num_nodes": self.num_nodes,
            "num_dofs": self.num_dofs,
            "num_edges": int(self.edges.shape[0]),
            "dx": self.dx,
            "dy": self.dy,
            "total_area": self.total_area,
            "min_coordinates": self.nodes.min(axis=0).tolist(),
            "max_coordinates": self.nodes.max(axis=0).tolist(),
        }


def _idx(i: int, j: int, nx: int) -> int:
    return j * nx + i


def _build_edges(nx: int, ny: int, include_diagonal: bool = True) -> np.ndarray:
    edges: List[Tuple[int, int]] = []
    for j in range(ny):
        for i in range(nx):
            a = _idx(i, j, nx)
            if i + 1 < nx:
                edges.append((a, _idx(i + 1, j, nx)))
            if j + 1 < ny:
                edges.append((a, _idx(i, j + 1, nx)))
            if include_diagonal and i + 1 < nx and j + 1 < ny:
                edges.append((a, _idx(i + 1, j + 1, nx)))
            if include_diagonal and i + 1 < nx and j - 1 >= 0:
                edges.append((a, _idx(i + 1, j - 1, nx)))
    return np.asarray(edges, dtype=int)


def _graph_laplacian(num_nodes: int, edges: np.ndarray):
    if sparse is not None:
        row = np.concatenate([edges[:, 0], edges[:, 1], edges[:, 0], edges[:, 1]])
        col = np.concatenate([edges[:, 1], edges[:, 0], edges[:, 0], edges[:, 1]])
        data = np.concatenate([-np.ones(len(edges)), -np.ones(len(edges)), np.ones(len(edges)), np.ones(len(edges))])
        return sparse.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    L = np.zeros((num_nodes, num_nodes), dtype=float)
    for a, b in edges:
        L[a, b] -= 1.0
        L[b, a] -= 1.0
        L[a, a] += 1.0
        L[b, b] += 1.0
    return L


def create_plate_mesh(Lx: float = 100.0, Ly: float = 80.0, nx: int = 31, ny: int = 25) -> PlateMesh:
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = float(Lx / (nx - 1))
    dy = float(Ly / (ny - 1))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
    wx = np.ones(nx)
    wy = np.ones(ny)
    wx[[0, -1]] = 0.5
    wy[[0, -1]] = 0.5
    weights_node = (np.outer(wy, wx).ravel() * dx * dy).astype(float)
    weights_dof = np.repeat(weights_node, 3)
    edges = _build_edges(nx, ny, include_diagonal=True)
    boundary_mask = np.zeros(nx * ny, dtype=bool)
    for j in range(ny):
        for i in range(nx):
            if i == 0 or j == 0 or i == nx - 1 or j == ny - 1:
                boundary_mask[_idx(i, j, nx)] = True
    return PlateMesh(Lx, Ly, nx, ny, nodes, edges, boundary_mask, weights_node, weights_dof, _graph_laplacian(nx * ny, edges), dx, dy)
