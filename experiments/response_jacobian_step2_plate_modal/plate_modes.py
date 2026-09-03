"""Analytical low-frequency modal basis for the Step 2 plate diagnostic."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from plate_mesh import PlateMesh

K = 8


def _raw_modes(mesh: PlateMesh) -> np.ndarray:
    X = mesh.nodes[:, 0]
    Y = mesh.nodes[:, 1]
    xn = 2.0 * X / mesh.Lx - 1.0
    yn = 2.0 * Y / mesh.Ly - 1.0
    sx1 = np.sin(np.pi * X / mesh.Lx)
    sx2 = np.sin(2.0 * np.pi * X / mesh.Lx)
    sy1 = np.sin(np.pi * Y / mesh.Ly)
    sy2 = np.sin(2.0 * np.pi * Y / mesh.Ly)

    modes = []
    for ux, uy, uz in [
        (0.0 * X, 0.0 * Y, sx1 * sy1),
        (0.0 * X, 0.0 * Y, sx2 * sy1),
        (0.0 * X, 0.0 * Y, sx1 * sy2),
        (0.0 * X, 0.0 * Y, 1.0 - xn**2),
        (0.0 * X, 0.0 * Y, 1.0 - yn**2),
        (xn, 0.0 * Y, 0.0 * X),
        (0.0 * X, yn, 0.0 * X),
        (0.0 * X, 0.0 * Y, xn * yn),
    ]:
        mode = np.column_stack([ux, uy, uz]).reshape(-1)
        modes.append(mode)
    return np.column_stack(modes).astype(float)


def mass_orthonormalize(V: np.ndarray, weights_dof: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    Q = []
    for j in range(V.shape[1]):
        v = V[:, j].copy()
        for q in Q:
            v -= q * float(np.dot(weights_dof * q, v))
        norm = float(np.sqrt(max(np.dot(weights_dof * v, v), 0.0)))
        if norm < tol:
            raise ValueError(f"Mode {j} is linearly dependent under mass inner product")
        Q.append(v / norm)
    return np.column_stack(Q)


def create_plate_modes(mesh: PlateMesh) -> Tuple[np.ndarray, Dict[str, object]]:
    raw = _raw_modes(mesh)
    B = mass_orthonormalize(raw, mesh.weights_dof)
    gram = B.T @ (mesh.weights_dof[:, None] * B)
    diagnostics = {
        "k": K,
        "orthonormality_error_fro": float(np.linalg.norm(gram - np.eye(K), ord="fro")),
        "gram_matrix": gram.tolist(),
        "mode_descriptions": [
            "z sin(pi x/Lx) sin(pi y/Ly)",
            "z sin(2pi x/Lx) sin(pi y/Ly)",
            "z sin(pi x/Lx) sin(2pi y/Ly)",
            "z x-normalized quadratic bowl",
            "z y-normalized quadratic bowl",
            "in-plane x shrink/stretch proportional to normalized x",
            "in-plane y shrink/stretch proportional to normalized y",
            "z twist proportional to normalized x * normalized y",
        ],
    }
    return B, diagnostics


def project_modal(B: np.ndarray, weights_dof: np.ndarray, displacement_flat: np.ndarray) -> np.ndarray:
    return B.T @ (weights_dof * displacement_flat)


def reconstruct_modal(B: np.ndarray, q: np.ndarray) -> np.ndarray:
    return B @ np.asarray(q, dtype=float)
