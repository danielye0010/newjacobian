"""Fixed triangular thin-plate mesh for Step 4 FEM-like diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy import sparse
except Exception:  # pragma: no cover
    sparse = None


@dataclass(frozen=True)
class FEMPlateMesh:
    Lx: float
    Ly: float
    nx: int
    ny: int
    nodes: np.ndarray
    triangles: np.ndarray
    edges: np.ndarray
    boundary_mask: np.ndarray
    weights_node: np.ndarray
    weights_dof: np.ndarray
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

    def triangle_areas(self, X: np.ndarray | None = None) -> np.ndarray:
        pts = self.nodes if X is None else X
        tri = pts[self.triangles]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return 0.5 * np.linalg.norm(cross, axis=1)

    def diagnostics(self) -> Dict[str, object]:
        areas = self.triangle_areas()
        return {
            "Lx": self.Lx,
            "Ly": self.Ly,
            "nx": self.nx,
            "ny": self.ny,
            "num_nodes": self.num_nodes,
            "num_triangles": int(self.triangles.shape[0]),
            "num_edges": int(self.edges.shape[0]),
            "num_dofs": self.num_dofs,
            "total_area": self.total_area,
            "min_triangle_area": float(np.min(areas)),
            "max_triangle_area": float(np.max(areas)),
            "min_coordinates": self.nodes.min(axis=0).tolist(),
            "max_coordinates": self.nodes.max(axis=0).tolist(),
        }


def _idx(i: int, j: int, nx: int) -> int:
    return j * nx + i


def _triangles(nx: int, ny: int) -> np.ndarray:
    tris: List[Tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = _idx(i, j, nx); b = _idx(i + 1, j, nx); c = _idx(i, j + 1, nx); d = _idx(i + 1, j + 1, nx)
            tris.append((a, b, d))
            tris.append((a, d, c))
    return np.asarray(tris, dtype=int)


def _edges_from_tris(tris: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b, c in tris:
        for u, v in [(a, b), (b, c), (c, a)]:
            if u > v: u, v = v, u
            edges.add((int(u), int(v)))
    return np.asarray(sorted(edges), dtype=int)


def create_fem_plate_mesh(Lx: float = 100.0, Ly: float = 80.0, nx: int = 25, ny: int = 21) -> FEMPlateMesh:
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = float(Lx / (nx - 1)); dy = float(Ly / (ny - 1))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
    tris = _triangles(nx, ny)
    edges = _edges_from_tris(tris)
    weights_node = np.zeros(nx * ny, dtype=float)
    tri_areas = 0.5 * dx * dy * np.ones(len(tris))
    for tri, area in zip(tris, tri_areas):
        weights_node[tri] += area / 3.0
    weights_dof = np.repeat(weights_node, 3)
    boundary = np.zeros(nx * ny, dtype=bool)
    for j in range(ny):
        for i in range(nx):
            if i == 0 or j == 0 or i == nx - 1 or j == ny - 1:
                boundary[_idx(i, j, nx)] = True
    return FEMPlateMesh(Lx, Ly, nx, ny, nodes, tris, edges, boundary, weights_node, weights_dof, dx, dy)
