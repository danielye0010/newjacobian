"""Lightweight FEM-like operators for fixed triangular plate meshes."""
from __future__ import annotations

import numpy as np

from fem_plate_mesh import FEMPlateMesh

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
except Exception:  # pragma: no cover
    sparse = None
    spsolve = None


def edge_weight_laplacian(mesh: FEMPlateMesh, X: np.ndarray, power: float = 1.0):
    edges = mesh.edges
    lengths = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
    weights = 1.0 / np.maximum(lengths, 1e-8) ** power
    n = mesh.num_nodes
    if sparse is not None:
        row = np.concatenate([edges[:, 0], edges[:, 1], edges[:, 0], edges[:, 1]])
        col = np.concatenate([edges[:, 1], edges[:, 0], edges[:, 0], edges[:, 1]])
        data = np.concatenate([-weights, -weights, weights, weights])
        return sparse.csr_matrix((data, (row, col)), shape=(n, n))
    L = np.zeros((n, n), dtype=float)
    for (a, b), w in zip(edges, weights):
        L[a, b] -= w; L[b, a] -= w; L[a, a] += w; L[b, b] += w
    return L


def stiffness_matrix(mesh: FEMPlateMesh, X: np.ndarray, kb: float, kt: float, anchor: float, eps: float = 1e-6):
    L = edge_weight_laplacian(mesh, X, power=1.0)
    n = mesh.num_nodes
    boundary = mesh.boundary_mask.astype(float)
    if sparse is not None:
        return (kb * (L.T @ L) + kt * L + sparse.diags(anchor * boundary + eps, format="csr")).tocsr()
    return kb * (L.T @ L) + kt * L + np.diag(anchor * boundary + eps)


def membrane_matrix(mesh: FEMPlateMesh, X: np.ndarray, kt: float, anchor: float, eps: float = 1e-6):
    L = edge_weight_laplacian(mesh, X, power=1.0)
    n = mesh.num_nodes
    boundary = mesh.boundary_mask.astype(float)
    if sparse is not None:
        return (kt * L + sparse.diags(anchor * boundary + eps, format="csr")).tocsr()
    return kt * L + np.diag(anchor * boundary + eps)


def solve_system(A, rhs: np.ndarray) -> np.ndarray:
    if sparse is not None and spsolve is not None and sparse.issparse(A):
        return np.asarray(spsolve(A, rhs), dtype=float)
    return np.linalg.solve(A, rhs)


def grid_slopes(mesh: FEMPlateMesh, z: np.ndarray):
    Z = z.reshape(mesh.ny, mesh.nx)
    dzdy, dzdx = np.gradient(Z, mesh.dy, mesh.dx, edge_order=2)
    return dzdx.ravel(), dzdy.ravel()
