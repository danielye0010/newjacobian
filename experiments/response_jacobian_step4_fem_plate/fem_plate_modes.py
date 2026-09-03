"""Analytical mass-normalized modal basis for Step 4."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from fem_plate_mesh import FEMPlateMesh

K = 8


def _raw_modes(mesh: FEMPlateMesh) -> np.ndarray:
    X = mesh.nodes[:, 0]; Y = mesh.nodes[:, 1]
    xn = 2.0 * X / mesh.Lx - 1.0; yn = 2.0 * Y / mesh.Ly - 1.0
    sx1 = np.sin(np.pi * X / mesh.Lx); sx2 = np.sin(2.0 * np.pi * X / mesh.Lx)
    sy1 = np.sin(np.pi * Y / mesh.Ly); sy2 = np.sin(2.0 * np.pi * Y / mesh.Ly)
    triples = [
        (0 * X, 0 * Y, sx1 * sy1),
        (0 * X, 0 * Y, sx2 * sy1),
        (0 * X, 0 * Y, sx1 * sy2),
        (0 * X, 0 * Y, 1.0 - xn**2),
        (0 * X, 0 * Y, 1.0 - yn**2),
        (xn, 0 * Y, 0 * X),
        (0 * X, yn, 0 * X),
        (0 * X, 0 * Y, xn * yn),
    ]
    return np.column_stack([np.column_stack(t).reshape(-1) for t in triples]).astype(float)


def mass_orthonormalize(V: np.ndarray, weights_dof: np.ndarray) -> np.ndarray:
    Q = []
    for j in range(V.shape[1]):
        v = V[:, j].copy()
        for q in Q:
            v -= q * float(np.dot(weights_dof * q, v))
        norm = float(np.sqrt(max(np.dot(weights_dof * v, v), 0.0)))
        if norm < 1e-12:
            raise ValueError(f"Dependent mode {j}")
        Q.append(v / norm)
    return np.column_stack(Q)


def create_fem_plate_modes(mesh: FEMPlateMesh) -> Tuple[np.ndarray, Dict[str, object]]:
    B = mass_orthonormalize(_raw_modes(mesh), mesh.weights_dof)
    gram = B.T @ (mesh.weights_dof[:, None] * B)
    return B, {
        "k": K,
        "orthonormality_error_fro": float(np.linalg.norm(gram - np.eye(K), ord="fro")),
        "gram_matrix": gram.tolist(),
        "mode_descriptions": [
            "z sin(pi x/Lx) sin(pi y/Ly)", "z sin(2pi x/Lx) sin(pi y/Ly)", "z sin(pi x/Lx) sin(2pi y/Ly)",
            "z quadratic bowl in x", "z quadratic bowl in y", "in-plane x shrink", "in-plane y shrink", "z twist xn*yn",
        ],
    }


def project_modal(B: np.ndarray, weights_dof: np.ndarray, displacement_flat: np.ndarray) -> np.ndarray:
    return B.T @ (weights_dof * displacement_flat)


def reconstruct_modal(B: np.ndarray, c: np.ndarray) -> np.ndarray:
    return B @ np.asarray(c, dtype=float)
