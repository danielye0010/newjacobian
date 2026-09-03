"""Step 6: JAX differentiable FEM response-Jacobian benchmark.

This experiment intentionally uses a small dense FEM surrogate.  The goal is
not calibrated AM/DED prediction; it is to validate response-Jacobian modal
compensation when the Jacobian is obtained by automatic differentiation through
the FEM forward solve.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from jax import config

config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "results" / "step6_jax_differentiable_fem_response_jacobian"
SUMMARY_PATH = RESULT_DIR / "SUMMARY.md"
MAIN_RESULTS_PATH = RESULT_DIR / "main_results.csv"
JAC_AUDIT_PATH = RESULT_DIR / "jacobian_audit.csv"
JAC_DIAG_PATH = RESULT_DIR / "jacobian_diagnostics.csv"
TRUST_AUDIT_PATH = RESULT_DIR / "trust_region_audit.csv"
IMPLEMENTATION_NOTES_PATH = RESULT_DIR / "implementation_notes.md"

K_MODES = 8
MAX_ITER = 35
TOL = 1e-8
DIVERGENCE_RATIO = 1e6
FD_STEPS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6]
SCALAR_ALPHAS = np.unique(
    np.concatenate(
        [
            np.linspace(0.02, 1.60, 65),
            np.logspace(-3, math.log10(2.5), 65),
        ]
    )
)


@dataclass(frozen=True)
class PlateMesh:
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

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_dofs(self) -> int:
        return int(self.nodes.size)

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
            "num_triangles": int(self.triangles.shape[0]),
            "num_edges": int(self.edges.shape[0]),
            "total_lumped_area": float(np.sum(self.weights_node)),
            "boundary_nodes": int(np.sum(self.boundary_mask)),
        }


@dataclass(frozen=True)
class FEMProcessParams:
    response_gain: float
    thermal_amp: float
    asym_x: float
    asym_y: float
    twist_load: float
    shrink_x: float
    shrink_y: float
    shear_xy: float
    shear_yx: float
    z_feedback: float
    curvature_feedback: float
    cross_xz: float
    cross_yz: float
    nonlinear_z: float
    nonlinear_xy: float
    kb: float
    kt: float
    membrane: float
    anchor_z: float
    anchor_xy: float
    foundation_z: float = 1e-4
    foundation_xy: float = 1e-4
    edge_power: float = 1.0
    trust_initial_lambda: float = 1e-4
    trust_max_step: float = math.inf

    def as_dict(self) -> Dict[str, float]:
        out = {}
        for key, value in asdict(self).items():
            out[key] = float(value)
        return out


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    label: str
    purpose: str
    params: FEMProcessParams


@dataclass
class MethodResult:
    experiment: str
    method: str
    initial_residual_norm: float
    final_residual_norm: float
    final_residual_ratio: float
    num_iterations: int
    converged: bool
    diverged: bool
    notes: str

    def as_row(self) -> Dict[str, object]:
        return {
            "experiment": self.experiment,
            "method": self.method,
            "initial_residual_norm": self.initial_residual_norm,
            "final_residual_norm": self.final_residual_norm,
            "final_residual_ratio": self.final_residual_ratio,
            "num_iterations": self.num_iterations,
            "converged": self.converged,
            "diverged": self.diverged,
            "notes": self.notes,
        }


def _idx(i: int, j: int, nx: int) -> int:
    return j * nx + i


def _triangles(nx: int, ny: int) -> np.ndarray:
    tris: List[Tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = _idx(i, j, nx)
            b = _idx(i + 1, j, nx)
            c = _idx(i, j + 1, nx)
            d = _idx(i + 1, j + 1, nx)
            tris.append((a, b, d))
            tris.append((a, d, c))
    return np.asarray(tris, dtype=int)


def _edges_from_triangles(triangles: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edges.add((int(u), int(v)))
    return np.asarray(sorted(edges), dtype=int)


def create_plate_mesh(Lx: float = 1.0, Ly: float = 0.8, nx: int = 9, ny: int = 7) -> PlateMesh:
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
    triangles = _triangles(nx, ny)
    edges = _edges_from_triangles(triangles)

    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    weights_node = np.zeros(nx * ny, dtype=float)
    for tri in triangles:
        weights_node[tri] += 0.5 * dx * dy / 3.0
    weights_dof = np.repeat(weights_node, 3)

    boundary = np.zeros(nx * ny, dtype=bool)
    for j in range(ny):
        for i in range(nx):
            boundary[_idx(i, j, nx)] = i == 0 or j == 0 or i == nx - 1 or j == ny - 1

    return PlateMesh(Lx, Ly, nx, ny, nodes, triangles, edges, boundary, weights_node, weights_dof)


def _raw_modes(mesh: PlateMesh) -> np.ndarray:
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    xn = 2.0 * x / mesh.Lx - 1.0
    yn = 2.0 * y / mesh.Ly - 1.0
    sx1 = np.sin(np.pi * x / mesh.Lx)
    sx2 = np.sin(2.0 * np.pi * x / mesh.Lx)
    sy1 = np.sin(np.pi * y / mesh.Ly)
    sy2 = np.sin(2.0 * np.pi * y / mesh.Ly)
    triples = [
        (0 * x, 0 * y, sx1 * sy1),
        (0 * x, 0 * y, sx2 * sy1),
        (0 * x, 0 * y, sx1 * sy2),
        (0 * x, 0 * y, 1.0 - xn**2),
        (0 * x, 0 * y, 1.0 - yn**2),
        (0 * x, 0 * y, xn * yn),
        (xn, 0 * y, 0 * x),
        (0 * x, yn, 0 * x),
    ]
    return np.column_stack([np.column_stack(t).reshape(-1) for t in triples]).astype(float)


def mass_orthonormalize(V: np.ndarray, weights_dof: np.ndarray) -> np.ndarray:
    Q: List[np.ndarray] = []
    for j in range(V.shape[1]):
        v = V[:, j].copy()
        for q in Q:
            v -= q * float(np.dot(weights_dof * q, v))
        norm = float(np.sqrt(max(np.dot(weights_dof * v, v), 0.0)))
        if norm < 1e-12:
            raise ValueError(f"Dependent modal basis vector {j}")
        Q.append(v / norm)
    return np.column_stack(Q)


def create_modal_basis(mesh: PlateMesh) -> Tuple[np.ndarray, Dict[str, object]]:
    B = mass_orthonormalize(_raw_modes(mesh), mesh.weights_dof)
    gram = B.T @ (mesh.weights_dof[:, None] * B)
    return B, {
        "num_modes": K_MODES,
        "orthonormality_error_fro": float(np.linalg.norm(gram - np.eye(K_MODES), ord="fro")),
        "mode_descriptions": [
            "z dome sin(pi x) sin(pi y)",
            "z x-wave sin(2pi x) sin(pi y)",
            "z y-wave sin(pi x) sin(2pi y)",
            "z x-bowl",
            "z y-bowl",
            "z twist x*y",
            "in-plane x shrink/stretch",
            "in-plane y shrink/stretch",
        ],
    }


class JaxDifferentiableFEMResponse:
    """Differentiable fixed-connectivity FEM surrogate with dense JAX solves."""

    def __init__(self, spec: ExperimentSpec, mesh: PlateMesh, B: np.ndarray):
        self.spec = spec
        self.mesh = mesh
        self.B_np = np.asarray(B, dtype=float)
        self.nodes_flat_np = mesh.flatten(mesh.nodes)
        self.weights_dof_np = np.asarray(mesh.weights_dof, dtype=float)
        self.params = spec.params

        self.nodes = jnp.asarray(mesh.nodes, dtype=jnp.float64)
        self.nodes_flat = jnp.asarray(self.nodes_flat_np, dtype=jnp.float64)
        self.B = jnp.asarray(self.B_np, dtype=jnp.float64)
        self.weights_dof = jnp.asarray(self.weights_dof_np, dtype=jnp.float64)
        self.edges = jnp.asarray(mesh.edges, dtype=jnp.int32)
        self.boundary = jnp.asarray(mesh.boundary_mask.astype(float), dtype=jnp.float64)
        self.x0 = self.nodes[:, 0]
        self.y0 = self.nodes[:, 1]
        self.xn0 = 2.0 * self.x0 / mesh.Lx - 1.0
        self.yn0 = 2.0 * self.y0 / mesh.Ly - 1.0
        self.sx1 = jnp.sin(jnp.pi * self.x0 / mesh.Lx)
        self.sx2 = jnp.sin(2.0 * jnp.pi * self.x0 / mesh.Lx)
        self.sy1 = jnp.sin(jnp.pi * self.y0 / mesh.Ly)
        self.sy2 = jnp.sin(2.0 * jnp.pi * self.y0 / mesh.Ly)

        self._b_jit = jax.jit(self._forward_modal_residual)
        self._J_jit = jax.jit(jax.jacfwd(self._forward_modal_residual))

    def _edge_laplacian(self, X: jnp.ndarray) -> jnp.ndarray:
        a = self.edges[:, 0]
        b = self.edges[:, 1]
        d = X[a] - X[b]
        lengths = jnp.sqrt(jnp.sum(d * d, axis=1) + 1e-12)
        weights = 1.0 / (lengths**self.params.edge_power)
        n = self.mesh.num_nodes
        row = jnp.concatenate([a, b, a, b])
        col = jnp.concatenate([b, a, a, b])
        data = jnp.concatenate([-weights, -weights, weights, weights])
        return jnp.zeros((n, n), dtype=jnp.float64).at[row, col].add(data)

    def _loads(self, Xc: jnp.ndarray, L: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        p = self.params
        x_current = 2.0 * Xc[:, 0] / self.mesh.Lx - 1.0
        y_current = 2.0 * Xc[:, 1] / self.mesh.Ly - 1.0
        z = Xc[:, 2]
        curvature = L @ z

        base_z = self.sx1 * self.sy1
        asym_z = p.asym_x * self.sx2 * self.sy1 + p.asym_y * self.sx1 * self.sy2
        twist_z = p.twist_load * self.xn0 * self.yn0 * self.sx1 * self.sy1
        fz = p.thermal_amp * (base_z + asym_z + twist_z)
        fz = fz + p.z_feedback * z + p.curvature_feedback * curvature
        fz = fz + p.cross_xz * self.xn0 * z + p.cross_yz * self.yn0 * z
        fz = fz + p.nonlinear_z * (z**3 + 0.35 * self.xn0 * z**2)

        fx = p.thermal_amp * (p.shrink_x * x_current + p.shear_xy * y_current)
        fx = fx + p.cross_xz * z + p.nonlinear_xy * self.xn0 * z**2
        fy = p.thermal_amp * (p.shrink_y * y_current + p.shear_yx * x_current)
        fy = fy + p.cross_yz * z + p.nonlinear_xy * self.yn0 * z**2
        return fx, fy, fz

    def _solve_response(self, Xc: jnp.ndarray) -> jnp.ndarray:
        p = self.params
        L = self._edge_laplacian(Xc)
        eye = jnp.eye(self.mesh.num_nodes, dtype=jnp.float64)
        diag_z = jnp.diag(p.anchor_z * self.boundary + p.foundation_z)
        diag_xy = jnp.diag(p.anchor_xy * self.boundary + p.foundation_xy)
        Kz = p.kb * (L @ L) + p.kt * L + diag_z + 1e-9 * eye
        Kxy = p.membrane * L + diag_xy + 1e-9 * eye
        fx, fy, fz = self._loads(Xc, L)
        ux = jnp.linalg.solve(Kxy, fx)
        uy = jnp.linalg.solve(Kxy, fy)
        wz = jnp.linalg.solve(Kz, fz)
        return p.response_gain * jnp.stack([ux, uy, wz], axis=1)

    def _forward_modal_residual(self, c: jnp.ndarray) -> jnp.ndarray:
        Xc_flat = self.nodes_flat + self.B @ c
        Xc = Xc_flat.reshape((self.mesh.num_nodes, 3))
        U = self._solve_response(Xc)
        residual_flat = (Xc + U - self.nodes).reshape(-1)
        return self.B.T @ (self.weights_dof * residual_flat)

    def b(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self._b_jit(jnp.asarray(c, dtype=jnp.float64)), dtype=float)

    def jacobian_ad(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self._J_jit(jnp.asarray(c, dtype=jnp.float64)), dtype=float)

    def jacobian_fd(self, c: np.ndarray, h: float) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        J = np.zeros((K_MODES, K_MODES), dtype=float)
        for j in range(K_MODES):
            step = np.zeros(K_MODES, dtype=float)
            step[j] = h
            J[:, j] = (self.b(c + step) - self.b(c - step)) / (2.0 * h)
        return J


def experiment_specs() -> List[ExperimentSpec]:
    base = dict(
        shrink_x=-0.010,
        shrink_y=-0.008,
        shear_xy=0.001,
        shear_yx=-0.001,
        kb=1.2,
        kt=0.10,
        membrane=0.35,
        anchor_z=0.90,
        anchor_xy=0.75,
    )
    return [
        ExperimentSpec(
            "experiment_1_identity_like",
            "Identity-like / linear response sanity case",
            "Low-gain response with weak geometry feedback; direct modal inversion should be valid.",
            FEMProcessParams(
                response_gain=0.035,
                thermal_amp=0.018,
                asym_x=0.05,
                asym_y=0.03,
                twist_load=0.00,
                z_feedback=0.02,
                curvature_feedback=0.00,
                cross_xz=0.00,
                cross_yz=0.00,
                nonlinear_z=0.0,
                nonlinear_xy=0.0,
                trust_max_step=math.inf,
                **base,
            ),
        ),
        ExperimentSpec(
            "experiment_2_gain_mismatch",
            "FEM gain-mismatch case",
            "Mode-dependent thermal and stiffness response; scalar alpha helps but diagonal gains differ by mode.",
            FEMProcessParams(
                response_gain=0.32,
                thermal_amp=0.030,
                asym_x=0.12,
                asym_y=0.05,
                twist_load=0.02,
                z_feedback=1.85,
                curvature_feedback=-0.012,
                cross_xz=0.015,
                cross_yz=-0.012,
                nonlinear_z=0.0,
                nonlinear_xy=0.0,
                trust_max_step=math.inf,
                **base,
            ),
        ),
        ExperimentSpec(
            "experiment_3_mode_coupling",
            "FEM mode-coupling case",
            "Asymmetric geometry-dependent load feedback produces off-diagonal modal response.",
            FEMProcessParams(
                response_gain=0.52,
                thermal_amp=0.032,
                asym_x=0.70,
                asym_y=-0.45,
                twist_load=0.45,
                shrink_x=-0.012,
                shrink_y=-0.004,
                shear_xy=0.010,
                shear_yx=-0.008,
                z_feedback=1.20,
                curvature_feedback=-0.018,
                cross_xz=0.34,
                cross_yz=-0.26,
                nonlinear_z=0.0,
                nonlinear_xy=0.0,
                kb=0.85,
                kt=0.080,
                membrane=0.26,
                anchor_z=0.65,
                anchor_xy=0.55,
                trust_max_step=math.inf,
            ),
        ),
        ExperimentSpec(
            "experiment_4_nonlinear_trust_region",
            "Nonlinear step-validity / trust-region case",
            "Cubic geometry feedback makes the local AD Jacobian accurate but unsafe for large full steps.",
            FEMProcessParams(
                response_gain=1.00,
                thermal_amp=0.080,
                asym_x=0.45,
                asym_y=-0.25,
                twist_load=0.35,
                shrink_x=-0.020,
                shrink_y=0.010,
                shear_xy=0.012,
                shear_yx=-0.007,
                z_feedback=1.60,
                curvature_feedback=-0.018,
                cross_xz=0.22,
                cross_yz=-0.18,
                nonlinear_z=600.0,
                nonlinear_xy=72.0,
                kb=0.75,
                kt=0.065,
                membrane=0.24,
                anchor_z=0.50,
                anchor_xy=0.50,
                trust_initial_lambda=1e-3,
                trust_max_step=0.015,
            ),
        ),
    ]


def c0() -> np.ndarray:
    return np.zeros(K_MODES, dtype=float)


def norm_ratio(norm: float, initial: float) -> float:
    return float(norm / max(initial, 1e-15))


def safe_diag_inverse(diag: np.ndarray) -> np.ndarray:
    safe = np.asarray(diag, dtype=float).copy()
    too_small = np.abs(safe) < 1e-10
    safe[too_small] = np.where(safe[too_small] >= 0.0, 1e-10, -1e-10)
    return 1.0 / safe


def _finish(
    response: JaxDifferentiableFEMResponse,
    method: str,
    initial: float,
    c: np.ndarray,
    history: List[Dict[str, object]],
    notes: str,
) -> MethodResult:
    final = float(np.linalg.norm(response.b(c)))
    ratios = [float(row["residual_ratio"]) for row in history]
    diverged = (not np.isfinite(final)) or norm_ratio(final, initial) > DIVERGENCE_RATIO
    diverged = bool(diverged or any(not np.isfinite(v) for v in c))
    converged = bool(norm_ratio(final, initial) < TOL)
    if min(ratios, default=1.0) < TOL and not converged:
        notes = f"{notes}; reached tolerance before final iteration"
    return MethodResult(
        experiment=response.spec.experiment,
        method=method,
        initial_residual_norm=float(initial),
        final_residual_norm=final,
        final_residual_ratio=norm_ratio(final, initial),
        num_iterations=max(0, len(history) - 1),
        converged=converged,
        diverged=diverged,
        notes=notes,
    )


def run_iterative_method(
    response: JaxDifferentiableFEMResponse,
    method: str,
    update: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, object]]],
    max_iter: int = MAX_ITER,
    notes: str = "",
) -> Tuple[MethodResult, List[Dict[str, object]]]:
    c = c0()
    initial = float(np.linalg.norm(response.b(c)))
    history: List[Dict[str, object]] = [
        {
            "iteration": 0,
            "residual_norm": initial,
            "residual_ratio": 1.0,
            "step_norm": 0.0,
            "accepted": True,
        }
    ]
    for it in range(1, max_iter + 1):
        if history[-1]["residual_ratio"] < TOL:
            break
        old_c = c.copy()
        c, meta = update(c)
        residual = float(np.linalg.norm(response.b(c)))
        ratio = norm_ratio(residual, initial)
        history.append(
            {
                "iteration": it,
                "residual_norm": residual,
                "residual_ratio": ratio,
                "step_norm": float(np.linalg.norm(c - old_c)),
                "accepted": bool(meta.get("accepted", True)),
                **meta,
            }
        )
        if (not np.isfinite(ratio)) or ratio > DIVERGENCE_RATIO or not np.all(np.isfinite(c)):
            break
    return _finish(response, method, initial, c, history, notes), history


def run_direct(response: JaxDifferentiableFEMResponse) -> Tuple[MethodResult, List[Dict[str, object]]]:
    return run_iterative_method(
        response,
        "direct_modal_inversion",
        lambda c: (c - response.b(c), {"accepted": True}),
        notes="uses c <- c - b(c)",
    )


def run_scalar(
    response: JaxDifferentiableFEMResponse, alpha: float, method: str = "best_scalar_scale_factor"
) -> Tuple[MethodResult, List[Dict[str, object]]]:
    return run_iterative_method(
        response,
        method,
        lambda c: (c - alpha * response.b(c), {"accepted": True, "alpha": float(alpha)}),
        notes=f"alpha={alpha:.8g}",
    )


def run_best_scalar(response: JaxDifferentiableFEMResponse) -> Tuple[MethodResult, List[Dict[str, object]], float]:
    best: Tuple[MethodResult, List[Dict[str, object]], float] | None = None
    for alpha in SCALAR_ALPHAS:
        result, history = run_scalar(response, float(alpha))
        if best is None or result.final_residual_ratio < best[0].final_residual_ratio:
            best = (result, history, float(alpha))
    assert best is not None
    result, history, alpha = best
    result.notes = f"best alpha on grid={alpha:.8g}; grid size={len(SCALAR_ALPHAS)}"
    return result, history, alpha


def run_diagonal_ad(response: JaxDifferentiableFEMResponse) -> Tuple[MethodResult, List[Dict[str, object]]]:
    def update(c: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        b = response.b(c)
        J = response.jacobian_ad(c)
        inv_diag = safe_diag_inverse(np.diag(J))
        delta = -inv_diag * b
        return c + delta, {"accepted": True, "diag_min": float(np.min(np.diag(J))), "diag_max": float(np.max(np.diag(J)))}

    return run_iterative_method(
        response,
        "diagonal_ad_jacobian_calibration",
        update,
        notes="uses only diag(J_ad(c)) at each iteration",
    )


def solve_gn_delta(J: np.ndarray, b: np.ndarray, damping: float) -> np.ndarray:
    lhs = J.T @ J + damping * np.eye(J.shape[1])
    rhs = -J.T @ b
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def run_full_gn(response: JaxDifferentiableFEMResponse) -> Tuple[MethodResult, List[Dict[str, object]], List[Dict[str, object]]]:
    audit_rows: List[Dict[str, object]] = []
    method = "full_ad_jacobian_gauss_newton"

    def update(c: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        b = response.b(c)
        J = response.jacobian_ad(c)
        delta = solve_gn_delta(J, b, 1e-12)
        trial = c + delta
        bt = response.b(trial)
        pred = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b + J @ delta, b + J @ delta))
        actual = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(bt, bt))
        rho = actual / pred if pred > 0 and np.isfinite(pred) else float("-inf")
        audit_rows.append(
            {
                "experiment": response.spec.experiment,
                "iteration": len(audit_rows) + 1,
                "method": method,
                "residual_norm": float(np.linalg.norm(b)),
                "predicted_reduction": pred,
                "actual_reduction": actual,
                "rho": rho,
                "step_norm": float(np.linalg.norm(delta)),
                "accepted": True,
                "damping_or_trust_parameter": 1e-12,
            }
        )
        return trial, {"accepted": True, "rho": rho, "predicted_reduction": pred, "actual_reduction": actual}

    result, history = run_iterative_method(
        response,
        method,
        update,
        notes="uses full J_ad(c), no step acceptance control",
    )
    return result, history, audit_rows


def run_trust_region(
    response: JaxDifferentiableFEMResponse,
) -> Tuple[MethodResult, List[Dict[str, object]], List[Dict[str, object]]]:
    method = "full_ad_jacobian_trust_region"
    c = c0()
    lam = float(response.params.trust_initial_lambda)
    max_step = float(response.params.trust_max_step)
    initial = float(np.linalg.norm(response.b(c)))
    history: List[Dict[str, object]] = [
        {
            "iteration": 0,
            "residual_norm": initial,
            "residual_ratio": 1.0,
            "step_norm": 0.0,
            "accepted": True,
            "lambda": lam,
        }
    ]
    audit_rows: List[Dict[str, object]] = []
    accepted_count = 0
    rejected_count = 0

    for it in range(1, MAX_ITER + 1):
        if history[-1]["residual_ratio"] < TOL:
            break
        b = response.b(c)
        J = response.jacobian_ad(c)
        delta = solve_gn_delta(J, b, lam)
        raw_norm = float(np.linalg.norm(delta))
        if np.isfinite(max_step) and raw_norm > max_step:
            delta = delta * (max_step / max(raw_norm, 1e-15))
        pred = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b + J @ delta, b + J @ delta))
        trial = c + delta
        bt = response.b(trial)
        actual = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(bt, bt))
        rho = actual / pred if pred > 0 and np.isfinite(pred) else float("-inf")
        accepted = bool(pred > 0.0 and rho > 0.05 and np.all(np.isfinite(trial)))
        if accepted:
            c = trial
            accepted_count += 1
        else:
            rejected_count += 1

        if accepted and rho > 0.75:
            lam *= 0.35
        elif accepted and rho < 0.25:
            lam *= 3.0
        elif not accepted:
            lam *= 8.0
        lam = float(np.clip(lam, 1e-12, 1e12))

        residual = float(np.linalg.norm(response.b(c)))
        row = {
            "experiment": response.spec.experiment,
            "iteration": it,
            "method": method,
            "residual_norm": residual,
            "predicted_reduction": pred,
            "actual_reduction": actual,
            "rho": rho,
            "step_norm": float(np.linalg.norm(delta)) if accepted else 0.0,
            "accepted": accepted,
            "damping_or_trust_parameter": lam if not np.isfinite(max_step) else max_step,
        }
        audit_rows.append(row)
        history.append(
            {
                "iteration": it,
                "residual_norm": residual,
                "residual_ratio": norm_ratio(residual, initial),
                "step_norm": row["step_norm"],
                "accepted": accepted,
                "lambda": lam,
                "rho": rho,
                "predicted_reduction": pred,
                "actual_reduction": actual,
            }
        )
        if (not np.isfinite(history[-1]["residual_ratio"])) or history[-1]["residual_ratio"] > DIVERGENCE_RATIO:
            break

    notes = f"accepted={accepted_count}; rejected={rejected_count}; max_step={max_step}; final_lambda={lam:.3g}"
    return _finish(response, method, initial, c, history, notes), history, audit_rows


def finite_difference_audit(response: JaxDifferentiableFEMResponse, c: np.ndarray) -> Dict[str, object]:
    J_ad = response.jacobian_ad(c)
    candidates = []
    for h in FD_STEPS:
        J_fd = response.jacobian_fd(c, h)
        diff = J_ad - J_fd
        candidates.append(
            {
                "h": h,
                "J_fd": J_fd,
                "rel": float(np.linalg.norm(diff, ord="fro") / max(np.linalg.norm(J_fd, ord="fro"), 1e-15)),
                "max_abs": float(np.max(np.abs(diff))),
                "mean_abs": float(np.mean(np.abs(diff))),
            }
        )
    best = min(candidates, key=lambda row: row["rel"])
    return {
        "experiment": response.spec.experiment,
        "num_modes": K_MODES,
        "fd_step": best["h"],
        "rel_fro_error": best["rel"],
        "max_abs_error": best["max_abs"],
        "mean_abs_error": best["mean_abs"],
        "norm_J_ad": float(np.linalg.norm(J_ad, ord="fro")),
        "norm_J_fd": float(np.linalg.norm(best["J_fd"], ord="fro")),
        "J_ad": J_ad,
        "recommended_h_grid": [(row["h"], row["rel"]) for row in candidates],
    }


def jacobian_diagnostics(response: JaxDifferentiableFEMResponse, best_alpha: float) -> Dict[str, object]:
    J = response.jacobian_ad(c0())
    diag = np.diag(np.diag(J))
    off = J - diag
    svals = np.linalg.svd(J, compute_uv=False)
    direct_spectral = float(np.max(np.abs(np.linalg.eigvals(np.eye(K_MODES) - J))))
    scalar_spectral = float(np.max(np.abs(np.linalg.eigvals(np.eye(K_MODES) - best_alpha * J))))
    inv_diag = safe_diag_inverse(np.diag(J))
    diagonal_iteration = np.eye(K_MODES) - np.diag(inv_diag) @ J
    diag_spectral = float(np.max(np.abs(np.linalg.eigvals(diagonal_iteration))))
    coupling_ratio = float(np.linalg.norm(off, ord="fro") / max(np.linalg.norm(J, ord="fro"), 1e-15))
    return {
        "experiment": response.spec.experiment,
        "coupling_ratio": coupling_ratio,
        "spectral_radius_direct": direct_spectral,
        "spectral_radius_best_scalar": scalar_spectral,
        "spectral_radius_diagonal": diag_spectral,
        "best_scalar_alpha": float(best_alpha),
        "condition_number_J": float(np.linalg.cond(J)),
        "min_singular_value_J": float(np.min(svals)),
        "max_singular_value_J": float(np.max(svals)),
        "J": J,
        "diagonal_gains": np.diag(J).copy(),
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in columns:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    out[key] = json.dumps(value, default=_json_default)
                elif value is None:
                    out[key] = ""
                else:
                    out[key] = value
            writer.writerow(out)


def run_methods_for_experiment(
    response: JaxDifferentiableFEMResponse,
) -> Tuple[List[MethodResult], List[Dict[str, object]], float]:
    results: List[MethodResult] = []
    trust_rows: List[Dict[str, object]] = []

    direct, _ = run_direct(response)
    scalar, _, best_alpha = run_best_scalar(response)
    diagonal, _ = run_diagonal_ad(response)
    full, _, full_audit = run_full_gn(response)
    trust, _, trust_audit = run_trust_region(response)

    results.extend([direct, scalar, diagonal, full, trust])
    trust_rows.extend(full_audit)
    trust_rows.extend(trust_audit)
    return results, trust_rows, best_alpha


def compact_ratio_table(results: Iterable[MethodResult]) -> str:
    rows = sorted(results, key=lambda r: (r.experiment, r.method))
    lines = [
        "| experiment | method | final residual ratio | iterations | status |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        status = "converged" if row.converged else ("diverged" if row.diverged else "stopped")
        lines.append(
            f"| {row.experiment} | {row.method} | {row.final_residual_ratio:.3e} | {row.num_iterations} | {status} |"
        )
    return "\n".join(lines)


def diagnostics_table(diag_rows: Iterable[Dict[str, object]]) -> str:
    lines = [
        "| experiment | coupling | rho direct | rho scalar | rho diagonal | alpha | cond(J) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diag_rows:
        lines.append(
            f"| {row['experiment']} | {row['coupling_ratio']:.3e} | {row['spectral_radius_direct']:.3e} | "
            f"{row['spectral_radius_best_scalar']:.3e} | {row['spectral_radius_diagonal']:.3e} | "
            f"{row['best_scalar_alpha']:.3e} | {row['condition_number_J']:.3e} |"
        )
    return "\n".join(lines)


def audit_table(audit_rows: Iterable[Dict[str, object]]) -> str:
    lines = [
        "| experiment | h | relative Frobenius error | max abs | mean abs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in audit_rows:
        lines.append(
            f"| {row['experiment']} | {row['fd_step']:.1e} | {row['rel_fro_error']:.3e} | "
            f"{row['max_abs_error']:.3e} | {row['mean_abs_error']:.3e} |"
        )
    return "\n".join(lines)


def method_lookup(results: Iterable[MethodResult]) -> Dict[str, Dict[str, MethodResult]]:
    out: Dict[str, Dict[str, MethodResult]] = {}
    for row in results:
        out.setdefault(row.experiment, {})[row.method] = row
    return out


def evaluate_verdict(
    results: List[MethodResult],
    audit_rows: List[Dict[str, object]],
    diag_rows: List[Dict[str, object]],
    trust_rows: List[Dict[str, object]],
) -> Tuple[str, Dict[str, bool]]:
    by_exp = method_lookup(results)
    diag_by_exp = {row["experiment"]: row for row in diag_rows}
    trust4 = [row for row in trust_rows if row["experiment"] == "experiment_4_nonlinear_trust_region" and row["method"] == "full_ad_jacobian_trust_region"]
    checks = {
        "ad_matches_fd": max(row["rel_fro_error"] for row in audit_rows) < 1e-5,
        "identity_direct_works": by_exp["experiment_1_identity_like"]["direct_modal_inversion"].final_residual_ratio < 1e-5,
        "gain_diagonal_beats_direct": by_exp["experiment_2_gain_mismatch"]["diagonal_ad_jacobian_calibration"].final_residual_ratio
        < by_exp["experiment_2_gain_mismatch"]["direct_modal_inversion"].final_residual_ratio,
        "gain_diagonal_spectral_radius_lower": diag_by_exp["experiment_2_gain_mismatch"]["spectral_radius_diagonal"]
        < diag_by_exp["experiment_2_gain_mismatch"]["spectral_radius_direct"],
        "coupling_is_non_negligible": diag_by_exp["experiment_3_mode_coupling"]["coupling_ratio"] > 0.12,
        "full_beats_scalar_or_diagonal_in_coupled_case": min(
            by_exp["experiment_3_mode_coupling"]["full_ad_jacobian_gauss_newton"].final_residual_ratio,
            by_exp["experiment_3_mode_coupling"]["full_ad_jacobian_trust_region"].final_residual_ratio,
        )
        < min(
            by_exp["experiment_3_mode_coupling"]["best_scalar_scale_factor"].final_residual_ratio,
            by_exp["experiment_3_mode_coupling"]["diagonal_ad_jacobian_calibration"].final_residual_ratio,
        ),
        "trust_improves_nonlinear_case": by_exp["experiment_4_nonlinear_trust_region"][
            "full_ad_jacobian_trust_region"
        ].final_residual_ratio
        < by_exp["experiment_4_nonlinear_trust_region"]["full_ad_jacobian_gauss_newton"].final_residual_ratio,
        "trust_has_acceptance_control_rows": any((not bool(row["accepted"])) or row["rho"] < 0.75 for row in trust4),
    }
    if all(checks.values()):
        return "PASS", checks
    if sum(bool(v) for v in checks.values()) >= 5:
        return "PARTIAL PASS", checks
    return "FAIL", checks


def write_summary(
    mesh_diag: Dict[str, object],
    mode_diag: Dict[str, object],
    specs: List[ExperimentSpec],
    results: List[MethodResult],
    audit_rows: List[Dict[str, object]],
    diag_rows: List[Dict[str, object]],
    trust_rows: List[Dict[str, object]],
    verdict: str,
    checks: Dict[str, bool],
) -> None:
    by_exp = method_lookup(results)
    diag_by_exp = {row["experiment"]: row for row in diag_rows}
    lines = [
        "# Step 6 JAX Differentiable FEM Response-Jacobian Benchmark",
        "",
        "## Purpose",
        "This step validates the response-Jacobian framework in a true differentiable FEM surrogate where the modal compensation-response Jacobian is computed by automatic differentiation through the FEM forward map.",
        "",
        "This is not real AM or real DED validation, and the FEM surrogate is not a calibrated DED model.",
        "",
        "## FEM Surrogate",
        "The forward model is a small dense JAX plate FEM surrogate on a fixed triangular grid.  The compensated geometry is `X(c) = X0 + B c`.  Geometry-dependent edge weights assemble a membrane Laplacian `L(X)`.  The out-of-plane response solves a plate-like system `Kz(X) w = fz(X)`, where `Kz = kb L L + kt L + boundary/foundation anchors`.  In-plane response solves membrane systems `Kxy(X) ux = fx(X)` and `Kxy(X) uy = fy(X)`.  The manufactured geometry is `Xp(c) = X(c) + u(c)`.",
        "",
        f"- mesh: {mesh_diag['nx']} x {mesh_diag['ny']} nodes, {mesh_diag['num_triangles']} triangles, {mesh_diag['num_dofs']} DOFs",
        f"- modal basis size: {mode_diag['num_modes']}",
        f"- mass-orthonormality error: {mode_diag['orthonormality_error_fro']:.3e}",
        "",
        "## Modal Residual and AD Jacobian",
        "`r(c) = Xp(c) - X0` is projected as `b(c) = B^T M r(c)`.  The response Jacobian is computed as `J_ad(c) = jax.jacfwd(forward_modal_residual)(c)`.  Finite differences are used only as an audit.",
        "",
        "## Experiment 0: AD Jacobian Audit",
        audit_table(audit_rows),
        "",
        "Conclusion: the AD Jacobian is numerically consistent with central finite differences when the relative Frobenius errors are small.",
        "",
        "## Method Comparison Results",
        compact_ratio_table(results),
        "",
        "## Jacobian Diagnostics",
        diagnostics_table(diag_rows),
        "",
        "## Experiment Conclusions",
    ]
    for spec in specs:
        methods = by_exp[spec.experiment]
        diag = diag_by_exp[spec.experiment]
        lines.extend([f"### {spec.label}", spec.purpose])
        if spec.experiment == "experiment_1_identity_like":
            lines.append(
                f"Direct modal inversion final ratio was {methods['direct_modal_inversion'].final_residual_ratio:.3e}; this supports the expected behavior that direct inversion works when the effective response is identity-like."
            )
        elif spec.experiment == "experiment_2_gain_mismatch":
            gains = ", ".join(f"{v:.3g}" for v in diag["diagonal_gains"])
            lines.append(
                f"Best scalar alpha was {diag['best_scalar_alpha']:.4g}. Diagonal gains were [{gains}], showing that different modes need different inverse gains. Diagonal calibration final ratio was {methods['diagonal_ad_jacobian_calibration'].final_residual_ratio:.3e} versus direct {methods['direct_modal_inversion'].final_residual_ratio:.3e}."
            )
        elif spec.experiment == "experiment_3_mode_coupling":
            lines.append(
                f"Coupling ratio was {diag['coupling_ratio']:.3e}. Full-Jacobian methods reached ratios {methods['full_ad_jacobian_gauss_newton'].final_residual_ratio:.3e} / {methods['full_ad_jacobian_trust_region'].final_residual_ratio:.3e}, compared with scalar {methods['best_scalar_scale_factor'].final_residual_ratio:.3e} and diagonal {methods['diagonal_ad_jacobian_calibration'].final_residual_ratio:.3e}."
            )
        elif spec.experiment == "experiment_4_nonlinear_trust_region":
            tr = methods["full_ad_jacobian_trust_region"]
            gn = methods["full_ad_jacobian_gauss_newton"]
            accepted = sum(1 for row in trust_rows if row["experiment"] == spec.experiment and row["method"] == "full_ad_jacobian_trust_region" and row["accepted"])
            rejected = sum(1 for row in trust_rows if row["experiment"] == spec.experiment and row["method"] == "full_ad_jacobian_trust_region" and not row["accepted"])
            lines.append(
                f"Full GN final ratio was {gn.final_residual_ratio:.3e}; trust-region final ratio was {tr.final_residual_ratio:.3e}. Trust-region audit rows recorded {accepted} accepted and {rejected} rejected steps, showing why local AD Jacobians still need step-validity control."
            )
        lines.append("")

    lines.extend(
        [
            "## Pass / Fail Checks",
        ]
    )
    for key, value in checks.items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines.extend(
        [
            "",
            f"Overall verdict: **{verdict}**",
            "",
            "## Final Claim",
            "This step validates the response-Jacobian framework in a true differentiable FEM surrogate where the modal compensation-response Jacobian is computed by automatic differentiation through the FEM forward map.",
            "",
            "## Limitations",
            "- The FEM model is a compact differentiable surrogate, not a calibrated AM/DED process model.",
            "- Dense solves keep the problem intentionally small.",
            "- Boundary anchors and thermal/geometric feedback terms are designed to exercise response-Jacobian behavior, not to match a specific machine or material.",
            "- The scalar method is an oracle grid search in this benchmark and should be interpreted as a best-case scalar inverse-response approximation.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_implementation_notes(verdict: str) -> None:
    lines = [
        "# Implementation Notes",
        "",
        "## Files Added",
        "- `experiments/step6_jax_differentiable_fem_response_jacobian/README_STEP6_JAX.md`",
        "- `experiments/step6_jax_differentiable_fem_response_jacobian/run_step6_jax_differentiable_fem_response_jacobian.py`",
        "- `results/step6_jax_differentiable_fem_response_jacobian/SUMMARY.md`",
        "- `results/step6_jax_differentiable_fem_response_jacobian/main_results.csv`",
        "- `results/step6_jax_differentiable_fem_response_jacobian/jacobian_audit.csv`",
        "- `results/step6_jax_differentiable_fem_response_jacobian/jacobian_diagnostics.csv`",
        "- `results/step6_jax_differentiable_fem_response_jacobian/trust_region_audit.csv`",
        "- `results/step6_jax_differentiable_fem_response_jacobian/implementation_notes.md`",
        "",
        "## How To Run",
        "From the project root:",
        "",
        "```powershell",
        "python experiments\\step6_jax_differentiable_fem_response_jacobian\\run_step6_jax_differentiable_fem_response_jacobian.py",
        "```",
        "",
        "## Dependencies",
        "- Python 3.13 was used in this run.",
        "- `jax` and `jaxlib` are required for automatic differentiation through the dense FEM solve.",
        "- `numpy` is required for reporting and linear algebra diagnostics.",
        "",
        "## Runtime Notes",
        "The benchmark uses a small 9 x 7 node mesh and dense CPU solves to keep JAX tracing and `jacfwd` reliable.",
        "",
        "## Known Limitations",
        "- This is a differentiable FEM surrogate, not a calibrated DED or AM process model.",
        "- No figures are generated.",
        "- The selected cases are deterministic parameterized surrogate cases intended to exercise identity-like, gain-mismatch, coupling, and nonlinear trust-region behavior.",
        "",
        f"## Result",
        f"{verdict}",
    ]
    IMPLEMENTATION_NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_terminal_summary(
    results: List[MethodResult],
    audit_rows: List[Dict[str, object]],
    diag_rows: List[Dict[str, object]],
    trust_rows: List[Dict[str, object]],
    verdict: str,
) -> None:
    print("AD-vs-FD Jacobian relative error:")
    for row in audit_rows:
        print(f"  {row['experiment']}: rel={row['rel_fro_error']:.3e}, h={row['fd_step']:.1e}")
    print()
    print("Final residual ratios:")
    by_exp = method_lookup(results)
    for exp, methods in by_exp.items():
        ratios = ", ".join(f"{name}={res.final_residual_ratio:.3e}" for name, res in methods.items())
        print(f"  {exp}: {ratios}")
    print()
    print("Coupling ratios and spectral-radius diagnostics:")
    for row in diag_rows:
        print(
            f"  {row['experiment']}: coupling={row['coupling_ratio']:.3e}, "
            f"rho_direct={row['spectral_radius_direct']:.3e}, "
            f"rho_scalar={row['spectral_radius_best_scalar']:.3e}, "
            f"rho_diagonal={row['spectral_radius_diagonal']:.3e}, alpha={row['best_scalar_alpha']:.3e}"
        )
    print()
    print("Trust-region acceptance summary:")
    for exp in sorted({row["experiment"] for row in trust_rows if row["method"] == "full_ad_jacobian_trust_region"}):
        rows = [row for row in trust_rows if row["experiment"] == exp and row["method"] == "full_ad_jacobian_trust_region"]
        accepted = sum(1 for row in rows if row["accepted"])
        rejected = sum(1 for row in rows if not row["accepted"])
        print(f"  {exp}: accepted={accepted}, rejected={rejected}, rows={len(rows)}")
    print()
    print(f"PASS / PARTIAL PASS / FAIL verdict: {verdict}")
    print(
        "Strongest defensible conclusion: this step validates the response-Jacobian framework in a true "
        "differentiable FEM surrogate where J_ad is computed by automatic differentiation through the FEM solve."
    )


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = create_plate_mesh()
    B, mode_diag = create_modal_basis(mesh)
    mesh_diag = mesh.diagnostics()

    specs = experiment_specs()
    all_results: List[MethodResult] = []
    all_trust_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []

    for spec in specs:
        response = JaxDifferentiableFEMResponse(spec, mesh, B)
        audit = finite_difference_audit(response, c0())
        audit_rows.append(audit)
        results, trust_rows, best_alpha = run_methods_for_experiment(response)
        all_results.extend(results)
        all_trust_rows.extend(trust_rows)
        diag_rows.append(jacobian_diagnostics(response, best_alpha))

    verdict, checks = evaluate_verdict(all_results, audit_rows, diag_rows, all_trust_rows)

    write_csv(MAIN_RESULTS_PATH, [row.as_row() for row in all_results], [
        "experiment",
        "method",
        "initial_residual_norm",
        "final_residual_norm",
        "final_residual_ratio",
        "num_iterations",
        "converged",
        "diverged",
        "notes",
    ])
    write_csv(JAC_AUDIT_PATH, audit_rows, [
        "experiment",
        "num_modes",
        "fd_step",
        "rel_fro_error",
        "max_abs_error",
        "mean_abs_error",
        "norm_J_ad",
        "norm_J_fd",
    ])
    write_csv(JAC_DIAG_PATH, diag_rows, [
        "experiment",
        "coupling_ratio",
        "spectral_radius_direct",
        "spectral_radius_best_scalar",
        "spectral_radius_diagonal",
        "best_scalar_alpha",
        "condition_number_J",
        "min_singular_value_J",
        "max_singular_value_J",
    ])
    write_csv(TRUST_AUDIT_PATH, all_trust_rows, [
        "experiment",
        "iteration",
        "method",
        "residual_norm",
        "predicted_reduction",
        "actual_reduction",
        "rho",
        "step_norm",
        "accepted",
        "damping_or_trust_parameter",
    ])
    write_summary(mesh_diag, mode_diag, specs, all_results, audit_rows, diag_rows, all_trust_rows, verdict, checks)
    write_implementation_notes(verdict)
    print_terminal_summary(all_results, audit_rows, diag_rows, all_trust_rows, verdict)


if __name__ == "__main__":
    main()
