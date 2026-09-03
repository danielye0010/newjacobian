"""Structured thin-plate mesh and lumped area weights for Step 2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class PlateMesh:
    Lx: float
    Ly: float
    nx: int
    ny: int
    nodes: np.ndarray
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

    def diagnostics(self) -> Dict[str, object]:
        return {
            "Lx": self.Lx,
            "Ly": self.Ly,
            "nx": self.nx,
            "ny": self.ny,
            "num_nodes": self.num_nodes,
            "num_dofs": self.num_dofs,
            "dx": self.dx,
            "dy": self.dy,
            "total_area": self.total_area,
            "min_coordinates": self.nodes.min(axis=0).tolist(),
            "max_coordinates": self.nodes.max(axis=0).tolist(),
        }


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
    return PlateMesh(Lx, Ly, nx, ny, nodes, weights_node, weights_dof, dx, dy)


def mass_inner(weights_dof: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(weights_dof * a, b))
