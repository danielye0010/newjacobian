"""Coupling-enhanced mesh-based physics response for Step 3B."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

from plate_mesh import PlateMesh
from plate_modes import K, project_modal, reconstruct_modal

try:
    from scipy import sparse
    from scipy.sparse.linalg import factorized
except Exception:  # pragma: no cover
    sparse = None
    factorized = None

C0 = np.zeros(K, dtype=float)
FD_EPS = 1e-5


@dataclass(frozen=True)
class ProcessParams:
    response_gain: float
    warp_amp: float
    z_gain: float
    slope_coupling_gain: float
    curvature_gain: float
    inplane_z_gain: float
    mixed_warp_gain: float
    mixed_xy_gain: float
    asymmetry: float
    smooth_strength: float
    sx: float = -0.006
    sy: float = 0.003
    shear_xy: float = 0.002
    shear_yx: float = -0.0015
    anchor_strength: float = 0.05
    smooth_x_scale: float = 0.7
    smooth_y_scale: float = 1.2
    smooth_z_scale: float = 0.35

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


class CouplingStressResponse:
    response_not_direct_modal_matrix = True

    def __init__(self, case_name: str, mesh: PlateMesh, B: np.ndarray, params: ProcessParams, fd_eps: float = FD_EPS):
        self.case_name = case_name
        self.mesh = mesh
        self.B = B
        self.params = params
        self.fd_eps = fd_eps
        self._solvers = self._build_solvers()

    def _build_solver(self, smooth_strength: float):
        n = self.mesh.num_nodes
        L = self.mesh.laplacian
        boundary = self.mesh.boundary_mask.astype(float)
        if sparse is not None and factorized is not None:
            A = sparse.eye(n, format="csc") + smooth_strength * (L.T @ L).tocsc()
            A = A + sparse.diags(self.params.anchor_strength * boundary, format="csc")
            return factorized(A.tocsc())
        A = np.eye(n) + smooth_strength * (L.T @ L) + np.diag(self.params.anchor_strength * boundary)
        return lambda rhs: np.linalg.solve(A, rhs)

    def _build_solvers(self):
        p = self.params
        base = p.smooth_strength
        return [
            self._build_solver(base * p.smooth_x_scale),
            self._build_solver(base * p.smooth_y_scale),
            self._build_solver(base * p.smooth_z_scale),
        ]

    def compensated_flat(self, c: np.ndarray) -> np.ndarray:
        return self.mesh.flatten(self.mesh.nodes) + reconstruct_modal(self.B, c)

    def _slopes_and_lap(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z = z.reshape(self.mesh.ny, self.mesh.nx)
        dzdy, dzdx = np.gradient(Z, self.mesh.dy, self.mesh.dx, edge_order=2)
        d2z_dx2 = np.gradient(dzdx, self.mesh.dx, axis=1, edge_order=2)
        d2z_dy2 = np.gradient(dzdy, self.mesh.dy, axis=0, edge_order=2)
        return dzdx.ravel(), dzdy.ravel(), (d2z_dx2 + d2z_dy2).ravel()

    def raw_distortion(self, Xc: np.ndarray) -> np.ndarray:
        p = self.params
        X0 = self.mesh.nodes
        x0 = X0[:, 0]
        y0 = X0[:, 1]
        x = Xc[:, 0] - self.mesh.Lx / 2.0
        y = Xc[:, 1] - self.mesh.Ly / 2.0
        xn = 2.0 * x0 / self.mesh.Lx - 1.0
        yn = 2.0 * y0 / self.mesh.Ly - 1.0
        zc = Xc[:, 2]
        uxc = Xc[:, 0] - X0[:, 0]
        uyc = Xc[:, 1] - X0[:, 1]
        dzdx, dzdy, lap_z = self._slopes_and_lap(zc)
        base = np.sin(np.pi * x0 / self.mesh.Lx) * np.sin(np.pi * y0 / self.mesh.Ly)
        asym = 0.5 * np.sin(2.0 * np.pi * x0 / self.mesh.Lx) * np.sin(np.pi * y0 / self.mesh.Ly)
        U = np.zeros_like(Xc)
        g = p.slope_coupling_gain
        U[:, 0] = p.sx * x + p.shear_xy * y + g * (0.9 * dzdx - 0.45 * dzdy) + p.mixed_xy_gain * yn * zc
        U[:, 1] = p.sy * y + p.shear_yx * x + g * (0.55 * dzdx + 0.85 * dzdy) - 0.8 * p.mixed_xy_gain * xn * zc
        U[:, 2] = (
            p.warp_amp * (base + p.asymmetry * asym)
            + p.z_gain * zc
            + p.curvature_gain * lap_z
            + p.inplane_z_gain * (uxc / self.mesh.Lx + 0.7 * uyc / self.mesh.Ly)
            + p.mixed_warp_gain * (xn * yn) * base
        )
        return U

    def physics_solve(self, Xc: np.ndarray) -> np.ndarray:
        raw = self.raw_distortion(Xc)
        U = np.column_stack([self._solvers[j](raw[:, j]) for j in range(3)])
        return self.params.response_gain * U

    def residual_flat(self, c: np.ndarray) -> np.ndarray:
        Xc = self.mesh.unflatten(self.compensated_flat(c))
        Xp = Xc + self.physics_solve(Xc)
        return self.mesh.flatten(Xp - self.mesh.nodes)

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


def diagnostics_for_response(response: CouplingStressResponse, basis_error: float) -> Dict[str, object]:
    J0 = response.jacobian_fd(C0)
    I = np.eye(K)
    eig = np.linalg.eigvals(J0)
    offdiag = J0 - np.diag(np.diag(J0))
    return {
        "J0": J0.tolist(),
        "eigenvalues": [[float(np.real(v)), float(np.imag(v))] for v in eig],
        "rho_I_minus_J0": float(np.max(np.abs(np.linalg.eigvals(I - J0)))),
        "identity_deviation": float(np.linalg.norm(J0 - I, ord="fro") / np.linalg.norm(I, ord="fro")),
        "coupling_ratio": float(np.linalg.norm(offdiag, ord="fro") / max(np.linalg.norm(J0, ord="fro"), 1e-15)),
        "condition_number": float(np.linalg.cond(J0)),
        "initial_residual_norm": float(np.linalg.norm(response.b(C0))),
        "finite_difference_eps": response.fd_eps,
        "basis_orthonormality_error_fro": float(basis_error),
        "process_parameters": response.params.as_dict(),
        "response_not_direct_modal_matrix": response.response_not_direct_modal_matrix,
    }

PhysicsPlateResponse = CouplingStressResponse
