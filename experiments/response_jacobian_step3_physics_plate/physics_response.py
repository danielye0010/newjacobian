"""Mesh-based physics-like plate response for Step 3.

The response originates from current node coordinates, slope feedback, and a fixed
Laplacian smoothing solve. It intentionally does not prescribe q_response = G @ c.
"""
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
    slope_gain: float
    asymmetry: float
    sx: float
    sy: float
    smooth_strength: float
    shear_xy: float = 0.0015
    shear_yx: float = -0.0010
    gxz: float = 0.18
    gyz: float = -0.12
    anchor_strength: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


class PhysicsPlateResponse:
    response_not_direct_modal_matrix = True

    def __init__(self, case_name: str, mesh: PlateMesh, B: np.ndarray, params: ProcessParams, fd_eps: float = FD_EPS):
        self.case_name = case_name
        self.mesh = mesh
        self.B = B
        self.params = params
        self.fd_eps = fd_eps
        self._solve = self._build_solver()

    def _build_solver(self):
        n = self.mesh.num_nodes
        L = self.mesh.laplacian
        boundary = self.mesh.boundary_mask.astype(float)
        if sparse is not None and factorized is not None:
            A = sparse.eye(n, format="csc") + self.params.smooth_strength * (L.T @ L).tocsc()
            if self.params.anchor_strength > 0:
                A = A + sparse.diags(self.params.anchor_strength * boundary, format="csc")
            return factorized(A.tocsc())
        A = np.eye(n) + self.params.smooth_strength * (L.T @ L)
        if self.params.anchor_strength > 0:
            A = A + np.diag(self.params.anchor_strength * boundary)
        return lambda rhs: np.linalg.solve(A, rhs)

    def compensated_flat(self, c: np.ndarray) -> np.ndarray:
        return self.mesh.flatten(self.mesh.nodes) + reconstruct_modal(self.B, c)

    def _slopes(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Z = z.reshape(self.mesh.ny, self.mesh.nx)
        dzdy, dzdx = np.gradient(Z, self.mesh.dy, self.mesh.dx, edge_order=2)
        return dzdx.ravel(), dzdy.ravel()

    def raw_distortion(self, Xc: np.ndarray) -> np.ndarray:
        p = self.params
        x0 = self.mesh.nodes[:, 0]
        y0 = self.mesh.nodes[:, 1]
        x = Xc[:, 0] - self.mesh.Lx / 2.0
        y = Xc[:, 1] - self.mesh.Ly / 2.0
        zc = Xc[:, 2]
        dzdx, dzdy = self._slopes(zc)
        x_phase = (x0 + 0.0) / self.mesh.Lx
        y_phase = (y0 + 0.0) / self.mesh.Ly
        base_warp = np.sin(np.pi * x_phase) * np.sin(np.pi * y_phase)
        asym_warp = 0.5 * np.sin(2.0 * np.pi * x_phase) * np.sin(np.pi * y_phase)
        slope_feedback = p.slope_gain * (dzdx**2 + dzdy**2)
        U = np.zeros_like(Xc)
        U[:, 0] = p.sx * x + p.shear_xy * y + p.gxz * zc
        U[:, 1] = p.sy * y + p.shear_yx * x + p.gyz * zc
        U[:, 2] = p.warp_amp * (base_warp + p.asymmetry * asym_warp) + p.z_gain * zc + slope_feedback
        return U

    def physics_solve(self, Xc: np.ndarray) -> np.ndarray:
        raw = self.raw_distortion(Xc)
        rhs = raw.copy()
        if self.params.anchor_strength > 0:
            rhs[self.mesh.boundary_mask, :] *= 1.0
        U = np.column_stack([self._solve(rhs[:, j]) for j in range(3)])
        return self.params.response_gain * U

    def residual_flat(self, c: np.ndarray) -> np.ndarray:
        Xc = self.mesh.unflatten(self.compensated_flat(c))
        U = self.physics_solve(Xc)
        Xp = Xc + U
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


def diagnostics_for_response(response: PhysicsPlateResponse, basis_error: float) -> Dict[str, object]:
    J0 = response.jacobian_fd(C0)
    I = np.eye(K)
    eigvals = np.linalg.eigvals(J0)
    offdiag = J0 - np.diag(np.diag(J0))
    return {
        "J0": J0.tolist(),
        "eigenvalues": [[float(np.real(v)), float(np.imag(v))] for v in eigvals],
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


def jax_status() -> Dict[str, object]:
    try:
        import jax  # noqa: F401
        return {"jax_available": True, "jax_used": False, "reason": "Finite difference was used because the response relies on scipy/numpy sparse factorization."}
    except Exception as exc:
        return {"jax_available": False, "jax_used": False, "reason": f"JAX import unavailable: {exc}"}
