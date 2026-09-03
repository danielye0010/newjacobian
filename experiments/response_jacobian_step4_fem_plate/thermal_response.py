"""FEM-like thermal plate response for Step 4."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from fem_operators import grid_slopes, membrane_matrix, solve_system, stiffness_matrix
from fem_plate_mesh import FEMPlateMesh
from fem_plate_modes import K, project_modal, reconstruct_modal

C0 = np.zeros(K, dtype=float)
FD_EPS = 1e-5


@dataclass(frozen=True)
class FEMProcessParams:
    response_gain: float
    thermal_amp: float
    asymmetry: float
    z_feedback: float
    slope_feedback: float
    sx: float
    sy: float
    kb: float
    kt: float
    anchor: float
    shear_xy: float = 0.0015
    shear_yx: float = -0.0010
    coupling_xz: float = 0.20
    coupling_yz: float = -0.12
    eps: float = 1e-5

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


class ThermalFEMResponse:
    response_is_fem_like_solve = True

    def __init__(self, case_name: str, mesh: FEMPlateMesh, B: np.ndarray, params: FEMProcessParams, fd_eps: float = FD_EPS):
        self.case_name = case_name
        self.mesh = mesh
        self.B = B
        self.params = params
        self.fd_eps = fd_eps

    def compensated_flat(self, c: np.ndarray) -> np.ndarray:
        return self.mesh.flatten(self.mesh.nodes) + reconstruct_modal(self.B, c)

    def thermal_loads(self, Xc: np.ndarray):
        p = self.params
        x0 = self.mesh.nodes[:, 0]
        y0 = self.mesh.nodes[:, 1]
        x = Xc[:, 0] - self.mesh.Lx / 2.0
        y = Xc[:, 1] - self.mesh.Ly / 2.0
        z = Xc[:, 2]
        dzdx, dzdy = grid_slopes(self.mesh, z)
        base = np.sin(np.pi * x0 / self.mesh.Lx) * np.sin(np.pi * y0 / self.mesh.Ly)
        asym = 0.5 * np.sin(2.0 * np.pi * x0 / self.mesh.Lx) * np.sin(np.pi * y0 / self.mesh.Ly)
        fz = p.thermal_amp * (base + p.asymmetry * asym) + p.z_feedback * z + p.slope_feedback * (dzdx**2 + dzdy**2)
        fx = p.sx * x + p.shear_xy * y + p.coupling_xz * z
        fy = p.sy * y + p.shear_yx * x + p.coupling_yz * z
        return fx, fy, fz

    def solve_response(self, Xc: np.ndarray) -> np.ndarray:
        p = self.params
        fx, fy, fz = self.thermal_loads(Xc)
        Kz = stiffness_matrix(self.mesh, Xc, p.kb, p.kt, p.anchor, p.eps)
        Kxy = membrane_matrix(self.mesh, Xc, max(p.kt, 1e-4), p.anchor, p.eps)
        ux = solve_system(Kxy, fx)
        uy = solve_system(Kxy, fy)
        w = solve_system(Kz, fz)
        return p.response_gain * np.column_stack([ux, uy, w])

    def residual_flat(self, c: np.ndarray) -> np.ndarray:
        Xc = self.mesh.unflatten(self.compensated_flat(c))
        U = self.solve_response(Xc)
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


def diagnostics_for_response(response: ThermalFEMResponse, basis_error: float) -> Dict[str, object]:
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
        "response_is_fem_like_solve": response.response_is_fem_like_solve,
    }


def jax_status() -> Dict[str, object]:
    try:
        import jax  # noqa: F401
        return {"jax_available": True, "jax_used": False, "reason": "Finite difference used because sparse FEM-like solves are scipy/numpy based."}
    except Exception as exc:
        return {"jax_available": False, "jax_used": False, "reason": f"JAX import unavailable: {exc}"}
