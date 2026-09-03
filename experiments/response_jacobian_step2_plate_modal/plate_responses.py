"""Geometry-space response models for the Step 2 plate modal diagnostic."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from plate_mesh import PlateMesh
from plate_modes import K, project_modal, reconstruct_modal

C0 = np.zeros(K, dtype=float)
Q0 = np.array([0.45, -0.35, 0.28, -0.22, 0.18, -0.12, 0.10, -0.08], dtype=float)


@dataclass(frozen=True)
class PlateResponse:
    case_name: str
    response_type: str
    mesh: PlateMesh
    B: np.ndarray
    beta: float
    G: np.ndarray
    q0: np.ndarray
    physical_u0: np.ndarray
    fd_eps: float = 1e-6

    def compensated_geometry_flat(self, c: np.ndarray) -> np.ndarray:
        return self.mesh.flatten(self.mesh.nodes) + reconstruct_modal(self.B, c)

    def residual_flat(self, c: np.ndarray) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        compensated = reconstruct_modal(self.B, c)
        if self.response_type == "frozen_response":
            return compensated + self.physical_u0
        c_geom = project_modal(self.B, self.mesh.weights_dof, compensated)
        q_response = self.G @ c_geom + self.beta * nonlinear_h(c_geom) + self.q0
        u_response = reconstruct_modal(self.B, q_response)
        return compensated + u_response

    def b(self, c: np.ndarray) -> np.ndarray:
        return project_modal(self.B, self.mesh.weights_dof, self.residual_flat(c))

    def jacobian_fd(self, c: np.ndarray, eps: float | None = None) -> np.ndarray:
        eps = self.fd_eps if eps is None else eps
        c = np.asarray(c, dtype=float)
        J = np.zeros((K, K), dtype=float)
        for i in range(K):
            step = np.zeros(K, dtype=float)
            step[i] = eps
            J[:, i] = (self.b(c + step) - self.b(c - step)) / (2.0 * eps)
        return J


def nonlinear_h(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    h = np.zeros(K, dtype=float)
    h[0] = c[0] ** 2
    h[1] = c[0] * c[1]
    h[2] = c[1] ** 2
    h[3] = c[2] * c[3]
    h[4] = c[4] ** 2 + 0.5 * c[0] * c[4]
    h[5] = c[5] * c[6]
    h[6] = c[2] ** 2
    h[7] = c[7] ** 2 + 0.25 * c[1] * c[7]
    return h


def physical_frozen_deformation(mesh: PlateMesh, B: np.ndarray, q0: np.ndarray) -> np.ndarray:
    X = mesh.nodes[:, 0]
    Y = mesh.nodes[:, 1]
    xn = 2.0 * X / mesh.Lx - 1.0
    yn = 2.0 * Y / mesh.Ly - 1.0
    extra = np.zeros_like(mesh.nodes)
    extra[:, 0] = -2.0e-4 * X
    extra[:, 1] = 1.5e-4 * Y
    extra[:, 2] = 0.015 * (1.0 - xn**2) * (1.0 - yn**2) + 0.006 * np.sin(3.0 * np.pi * X / mesh.Lx) * np.sin(np.pi * Y / mesh.Ly)
    return reconstruct_modal(B, q0) + extra.reshape(-1)


def _orthogonal(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(K, K))
    Q, R = np.linalg.qr(G)
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    return Q * signs


def _target_jacobians() -> Dict[str, np.ndarray]:
    easy = np.diag([0.62, 0.82, 1.05, 1.18, 0.92, 0.72, 1.30, 1.00])
    marginal = np.diag([0.20, 0.65, 1.10, 1.80, 2.00, 0.90, 1.20, 1.00])
    hard_gain = np.diag([0.30, 0.80, 1.20, 2.40, 2.80, 0.50, 1.50, 2.20])
    Q = _orthogonal(202)
    hard_coupled = Q @ np.diag([0.40, 0.90, 1.40, 2.50, 3.00, 0.70, 1.80, 2.20]) @ Q.T
    return {
        "geom_easy": easy,
        "geom_marginal": marginal,
        "geom_hard_gain": hard_gain,
        "geom_hard_coupled_nonlinear": hard_coupled,
    }


def make_responses(mesh: PlateMesh, B: np.ndarray) -> Dict[str, PlateResponse]:
    zero_G = np.zeros((K, K), dtype=float)
    physical = physical_frozen_deformation(mesh, B, Q0)
    responses: Dict[str, PlateResponse] = {
        "frozen_response": PlateResponse("frozen_response", "frozen_response", mesh, B, 0.0, zero_G, Q0, physical),
    }
    for name, J_target in _target_jacobians().items():
        beta = 0.15 if name == "geom_hard_coupled_nonlinear" else 0.0
        responses[name] = PlateResponse(name, "geometry_dependent_response", mesh, B, beta, J_target - np.eye(K), Q0, np.zeros(mesh.num_dofs))
    return responses


def response_diagnostics(response: PlateResponse, basis_orthonormality_error: float) -> Dict[str, object]:
    J0 = response.jacobian_fd(C0)
    I = np.eye(K)
    eigvals = np.linalg.eigvals(J0)
    offdiag = J0 - np.diag(np.diag(J0))
    return {
        "response_type": response.response_type,
        "beta": response.beta,
        "J0": J0.tolist(),
        "eigenvalues": [[float(np.real(v)), float(np.imag(v))] for v in eigvals],
        "rho_I_minus_J0": float(np.max(np.abs(np.linalg.eigvals(I - J0)))),
        "identity_deviation": float(np.linalg.norm(J0 - I, ord="fro") / np.linalg.norm(I, ord="fro")),
        "coupling_ratio": float(np.linalg.norm(offdiag, ord="fro") / max(np.linalg.norm(J0, ord="fro"), 1e-15)),
        "condition_number": float(np.linalg.cond(J0)),
        "initial_residual_norm": float(np.linalg.norm(response.b(C0))),
        "basis_orthonormality_error_fro": float(basis_orthonormality_error),
        "finite_difference_eps": response.fd_eps,
    }
