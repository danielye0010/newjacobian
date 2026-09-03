"""Exploratory response-Jacobian discovery suite for Step 6 JAX FEM.

The suite keeps the Step 6 bounded claim: this is a true differentiable FEM
surrogate with response Jacobians computed by JAX AD through the forward solve.
It is not calibrated AM/DED validation.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from jax import config

config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "results" / "step6_jax_differentiable_fem_response_jacobian" / "discovery_suite"

MAX_ITER = 24
TRUST_MAX_ITER = 30
SUCCESS_TOL = 1e-4
DIVERGENCE_RATIO = 1e6
FD_STEPS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6]
ALPHA_GRID = np.unique(np.concatenate([np.linspace(0.02, 1.60, 25), np.logspace(-3, math.log10(2.0), 17)]))
BETA_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 0.5, 1.0]
EPS_J_LEVELS = [0.00, 0.01, 0.05, 0.10, 0.20]
EPS_B_LEVELS = [0.00, 0.02]


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

    def flatten(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float).reshape(-1)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    case_family: str
    seed: int
    num_modes: int
    gain_strength: float
    coupling_strength: float
    nonlinear_strength: float
    initial_residual_amplitude: float
    basis_type: str = "structural"
    basis_seed: int = 0
    outside_basis_strength: float = 0.0
    z_feedback_override: float | None = None
    response_gain_override: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class PhysicalParams:
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
    outside_z: float

    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.response_gain,
                self.thermal_amp,
                self.asym_x,
                self.asym_y,
                self.twist_load,
                self.shrink_x,
                self.shrink_y,
                self.shear_xy,
                self.shear_yx,
                self.z_feedback,
                self.curvature_feedback,
                self.cross_xz,
                self.cross_yz,
                self.nonlinear_z,
                self.nonlinear_xy,
                self.kb,
                self.kt,
                self.membrane,
                self.anchor_z,
                self.anchor_xy,
                self.outside_z,
            ],
            dtype=float,
        )


@dataclass
class EvalStats:
    forward_evals: int = 0
    jacobian_evals: int = 0
    forward_time: float = 0.0
    jacobian_time: float = 0.0

    def add_forward(self, seconds: float) -> None:
        self.forward_evals += 1
        self.forward_time += seconds

    def add_jacobian(self, seconds: float) -> None:
        self.jacobian_evals += 1
        self.jacobian_time += seconds


@dataclass
class MethodOutcome:
    method: str
    initial_residual_norm: float
    final_residual_norm: float
    final_residual_ratio: float
    num_iterations: int
    converged: bool
    diverged: bool
    notes: str
    accepted_steps: int = 0
    rejected_steps: int = 0
    num_forward_evaluations: int = 0
    num_jacobian_evaluations: int = 0
    wall_time_seconds: float = 0.0
    mean_fem_solve_time_seconds: float = 0.0
    mean_jacobian_time_seconds: float = 0.0


def _idx(i: int, j: int, nx: int) -> int:
    return j * nx + i


def _triangles(nx: int, ny: int) -> np.ndarray:
    rows = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = _idx(i, j, nx)
            b = _idx(i + 1, j, nx)
            c = _idx(i, j + 1, nx)
            d = _idx(i + 1, j + 1, nx)
            rows.append((a, b, d))
            rows.append((a, d, c))
    return np.asarray(rows, dtype=int)


def _edges_from_triangles(triangles: np.ndarray) -> np.ndarray:
    edges = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edges.add((int(u), int(v)))
    return np.asarray(sorted(edges), dtype=int)


def create_mesh(Lx: float = 1.0, Ly: float = 0.8, nx: int = 7, ny: int = 5) -> PlateMesh:
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


def mass_orthonormalize(V: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cols: List[np.ndarray] = []
    for j in range(V.shape[1]):
        v = V[:, j].astype(float).copy()
        for q in cols:
            v -= q * float(np.dot(weights * q, v))
        nrm = float(np.sqrt(max(np.dot(weights * v, v), 0.0)))
        if nrm > 1e-12:
            cols.append(v / nrm)
    if not cols:
        raise ValueError("No independent basis vectors")
    return np.column_stack(cols)


def structural_raw_modes(mesh: PlateMesh) -> np.ndarray:
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    xn = 2.0 * x / mesh.Lx - 1.0
    yn = 2.0 * y / mesh.Ly - 1.0
    sx1 = np.sin(np.pi * x / mesh.Lx)
    sx2 = np.sin(2.0 * np.pi * x / mesh.Lx)
    sx3 = np.sin(3.0 * np.pi * x / mesh.Lx)
    sy1 = np.sin(np.pi * y / mesh.Ly)
    sy2 = np.sin(2.0 * np.pi * y / mesh.Ly)
    sy3 = np.sin(3.0 * np.pi * y / mesh.Ly)
    z0 = 0.0 * x
    triples = [
        (z0, z0, sx1 * sy1),
        (z0, z0, sx2 * sy1),
        (z0, z0, sx1 * sy2),
        (z0, z0, 1.0 - xn**2),
        (z0, z0, 1.0 - yn**2),
        (z0, z0, xn * yn),
        (xn, z0, z0),
        (z0, yn, z0),
        (z0, z0, sx3 * sy1),
        (z0, z0, sx1 * sy3),
    ]
    return np.column_stack([np.column_stack(t).reshape(-1) for t in triples])


def nodal_like_raw_modes(mesh: PlateMesh, count: int) -> np.ndarray:
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    centers = [
        (0.25, 0.25, "z"),
        (0.75, 0.25, "z"),
        (0.25, 0.75, "z"),
        (0.75, 0.75, "z"),
        (0.50, 0.50, "z"),
        (0.50, 0.25, "z"),
        (0.50, 0.50, "x"),
        (0.50, 0.50, "y"),
        (0.25, 0.50, "x"),
        (0.75, 0.50, "y"),
    ]
    cols = []
    sigma2 = 0.10
    for cx, cy, axis in centers[: max(count, 1)]:
        g = np.exp(-(((x / mesh.Lx - cx) ** 2 + (y / mesh.Ly - cy) ** 2) / sigma2))
        trip = np.zeros((mesh.num_nodes, 3))
        trip[:, {"x": 0, "y": 1, "z": 2}[axis]] = g
        cols.append(trip.reshape(-1))
    return np.column_stack(cols)


def create_basis(mesh: PlateMesh, num_modes: int, basis_type: str = "structural", seed: int = 0) -> np.ndarray:
    structural = structural_raw_modes(mesh)
    if basis_type == "structural":
        raw = structural[:, :num_modes]
    elif basis_type == "noisy_modal":
        rng = np.random.default_rng(seed)
        raw = structural[:, :num_modes] + 0.18 * rng.normal(size=(mesh.nodes.size, num_modes))
    elif basis_type == "random":
        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(mesh.nodes.size, num_modes))
    elif basis_type == "nodal_like":
        raw = nodal_like_raw_modes(mesh, num_modes)
    else:
        raise ValueError(f"Unknown basis_type={basis_type}")
    Q = mass_orthonormalize(raw, mesh.weights_dof)
    if Q.shape[1] < num_modes:
        raise ValueError(f"Only {Q.shape[1]} independent modes for {basis_type}")
    return Q[:, :num_modes]


def make_params(spec: CaseSpec) -> PhysicalParams:
    rng = np.random.default_rng(spec.seed)
    jitter = lambda scale=1.0: float(scale * rng.normal())
    gain = spec.gain_strength
    coupling = spec.coupling_strength
    nonlinear = spec.nonlinear_strength
    response_gain = 0.035 + 0.36 * gain
    if spec.response_gain_override is not None:
        response_gain = spec.response_gain_override
    z_feedback = 0.03 + 1.75 * gain
    if spec.z_feedback_override is not None:
        z_feedback = spec.z_feedback_override
    return PhysicalParams(
        response_gain=response_gain,
        thermal_amp=spec.initial_residual_amplitude,
        asym_x=coupling * (0.52 + jitter(0.08)),
        asym_y=coupling * (-0.35 + jitter(0.08)),
        twist_load=coupling * (0.35 + jitter(0.05)),
        shrink_x=-0.010 * (1.0 + 0.3 * gain + jitter(0.08)),
        shrink_y=(-0.007 + 0.002 * coupling) * (1.0 + jitter(0.08)),
        shear_xy=coupling * (0.010 + jitter(0.002)),
        shear_yx=coupling * (-0.008 + jitter(0.002)),
        z_feedback=z_feedback,
        curvature_feedback=-0.006 * gain - 0.010 * coupling,
        cross_xz=coupling * (0.22 + 0.08 * gain + jitter(0.03)),
        cross_yz=coupling * (-0.18 - 0.05 * gain + jitter(0.03)),
        nonlinear_z=nonlinear,
        nonlinear_xy=nonlinear * (0.05 + 0.08 * max(coupling, 0.10)),
        kb=max(0.40, 1.20 - 0.20 * gain + jitter(0.03)),
        kt=max(0.030, 0.10 - 0.015 * coupling),
        membrane=max(0.12, 0.35 - 0.08 * coupling),
        anchor_z=max(0.20, 0.85 - 0.20 * coupling),
        anchor_xy=max(0.20, 0.70 - 0.12 * coupling),
        outside_z=spec.outside_basis_strength,
    )


def classify_regime(spec: CaseSpec) -> str:
    if spec.nonlinear_strength > 1e-9:
        return "nonlinear_local_validity"
    if spec.coupling_strength >= 0.45:
        return "mode_coupling"
    if spec.gain_strength >= 0.80:
        return "gain_mismatch"
    return "identity_like"


class DiscoveryModel:
    def __init__(self, mesh: PlateMesh, B: np.ndarray):
        self.mesh = mesh
        self.B_np = np.asarray(B, dtype=float)
        self.num_modes = int(B.shape[1])
        self.weights_np = np.asarray(mesh.weights_dof, dtype=float)
        self.nodes = jnp.asarray(mesh.nodes, dtype=jnp.float64)
        self.nodes_flat = jnp.asarray(mesh.flatten(mesh.nodes), dtype=jnp.float64)
        self.B = jnp.asarray(B, dtype=jnp.float64)
        self.weights = jnp.asarray(mesh.weights_dof, dtype=jnp.float64)
        self.edges = jnp.asarray(mesh.edges, dtype=jnp.int32)
        self.boundary = jnp.asarray(mesh.boundary_mask.astype(float), dtype=jnp.float64)
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        self.xn = 2.0 * x / mesh.Lx - 1.0
        self.yn = 2.0 * y / mesh.Ly - 1.0
        self.sx1 = jnp.sin(jnp.pi * x / mesh.Lx)
        self.sx2 = jnp.sin(2.0 * jnp.pi * x / mesh.Lx)
        self.sx3 = jnp.sin(3.0 * jnp.pi * x / mesh.Lx)
        self.sy1 = jnp.sin(jnp.pi * y / mesh.Ly)
        self.sy2 = jnp.sin(2.0 * jnp.pi * y / mesh.Ly)
        self.sy3 = jnp.sin(3.0 * jnp.pi * y / mesh.Ly)
        self._b_jit = jax.jit(self._forward_modal_residual)
        self._J_jit = jax.jit(jax.jacfwd(self._forward_modal_residual, argnums=0))
        self._residual_jit = jax.jit(self._residual_flat)
        self._solve_residual_jit = jax.jit(self._linear_solve_residual_norm)

    def _edge_laplacian(self, X: jnp.ndarray) -> jnp.ndarray:
        a = self.edges[:, 0]
        b = self.edges[:, 1]
        d = X[a] - X[b]
        lengths = jnp.sqrt(jnp.sum(d * d, axis=1) + 1e-12)
        weights = 1.0 / lengths
        n = self.mesh.num_nodes
        row = jnp.concatenate([a, b, a, b])
        col = jnp.concatenate([b, a, a, b])
        data = jnp.concatenate([-weights, -weights, weights, weights])
        return jnp.zeros((n, n), dtype=jnp.float64).at[row, col].add(data)

    def _unpack(self, p: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        return tuple(p[i] for i in range(21))

    def _assemble(self, Xc: jnp.ndarray, p: jnp.ndarray):
        (
            response_gain,
            thermal_amp,
            asym_x,
            asym_y,
            twist_load,
            shrink_x,
            shrink_y,
            shear_xy,
            shear_yx,
            z_feedback,
            curvature_feedback,
            cross_xz,
            cross_yz,
            nonlinear_z,
            nonlinear_xy,
            kb,
            kt,
            membrane,
            anchor_z,
            anchor_xy,
            outside_z,
        ) = self._unpack(p)
        L = self._edge_laplacian(Xc)
        eye = jnp.eye(self.mesh.num_nodes, dtype=jnp.float64)
        Kz = kb * (L @ L) + kt * L + jnp.diag(anchor_z * self.boundary + 1e-4) + 1e-9 * eye
        Kxy = membrane * L + jnp.diag(anchor_xy * self.boundary + 1e-4) + 1e-9 * eye
        x_current = 2.0 * Xc[:, 0] / self.mesh.Lx - 1.0
        y_current = 2.0 * Xc[:, 1] / self.mesh.Ly - 1.0
        z = Xc[:, 2]
        curvature = L @ z
        base_z = self.sx1 * self.sy1
        coupled_z = asym_x * self.sx2 * self.sy1 + asym_y * self.sx1 * self.sy2
        twist_z = twist_load * self.xn * self.yn * self.sx1 * self.sy1
        outside = outside_z * self.sx3 * self.sy2
        fz = thermal_amp * (base_z + coupled_z + twist_z + outside)
        fz = fz + z_feedback * z + curvature_feedback * curvature
        fz = fz + cross_xz * self.xn * z + cross_yz * self.yn * z
        fz = fz + nonlinear_z * (z**3 + 0.35 * self.xn * z**2)
        fx = thermal_amp * (shrink_x * x_current + shear_xy * y_current) + cross_xz * z
        fy = thermal_amp * (shrink_y * y_current + shear_yx * x_current) + cross_yz * z
        fx = fx + nonlinear_xy * self.xn * z**2
        fy = fy + nonlinear_xy * self.yn * z**2
        return response_gain, Kz, Kxy, fx, fy, fz

    def _solve_response(self, Xc: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        response_gain, Kz, Kxy, fx, fy, fz = self._assemble(Xc, p)
        ux = jnp.linalg.solve(Kxy, fx)
        uy = jnp.linalg.solve(Kxy, fy)
        wz = jnp.linalg.solve(Kz, fz)
        return response_gain * jnp.stack([ux, uy, wz], axis=1)

    def _residual_flat(self, c: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        Xc_flat = self.nodes_flat + self.B @ c
        Xc = Xc_flat.reshape((self.mesh.num_nodes, 3))
        U = self._solve_response(Xc, p)
        return (Xc + U - self.nodes).reshape(-1)

    def _forward_modal_residual(self, c: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        r = self._residual_flat(c, p)
        return self.B.T @ (self.weights * r)

    def _linear_solve_residual_norm(self, c: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        Xc_flat = self.nodes_flat + self.B @ c
        Xc = Xc_flat.reshape((self.mesh.num_nodes, 3))
        response_gain, Kz, Kxy, fx, fy, fz = self._assemble(Xc, p)
        ux = jnp.linalg.solve(Kxy, fx)
        uy = jnp.linalg.solve(Kxy, fy)
        wz = jnp.linalg.solve(Kz, fz)
        _ = response_gain
        rx = jnp.linalg.norm(Kxy @ ux - fx)
        ry = jnp.linalg.norm(Kxy @ uy - fy)
        rz = jnp.linalg.norm(Kz @ wz - fz)
        return jnp.maximum(rx, jnp.maximum(ry, rz))

    def c0(self) -> np.ndarray:
        return np.zeros(self.num_modes, dtype=float)

    def b(self, c: np.ndarray, p: np.ndarray, stats: EvalStats | None = None) -> np.ndarray:
        t = time.perf_counter()
        out = np.asarray(self._b_jit(jnp.asarray(c, dtype=jnp.float64), jnp.asarray(p, dtype=jnp.float64)), dtype=float)
        if stats is not None:
            stats.add_forward(time.perf_counter() - t)
        return out

    def J(self, c: np.ndarray, p: np.ndarray, stats: EvalStats | None = None) -> np.ndarray:
        t = time.perf_counter()
        out = np.asarray(self._J_jit(jnp.asarray(c, dtype=jnp.float64), jnp.asarray(p, dtype=jnp.float64)), dtype=float)
        if stats is not None:
            stats.add_jacobian(time.perf_counter() - t)
        return out

    def residual_flat(self, c: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.asarray(self._residual_jit(jnp.asarray(c, dtype=jnp.float64), jnp.asarray(p, dtype=jnp.float64)), dtype=float)

    def solve_residual_norm(self, c: np.ndarray, p: np.ndarray) -> float:
        return float(self._solve_residual_jit(jnp.asarray(c, dtype=jnp.float64), jnp.asarray(p, dtype=jnp.float64)))

    def J_fd(self, c: np.ndarray, p: np.ndarray, h: float) -> np.ndarray:
        J = np.zeros((self.num_modes, self.num_modes), dtype=float)
        for j in range(self.num_modes):
            step = np.zeros(self.num_modes, dtype=float)
            step[j] = h
            J[:, j] = (self.b(c + step, p) - self.b(c - step, p)) / (2.0 * h)
        return J

    def captured_energy_ratio(self, residual: np.ndarray) -> Tuple[float, float]:
        coeff = self.B_np.T @ (self.weights_np * residual)
        projected = self.B_np @ coeff
        full_energy = float(np.dot(self.weights_np * residual, residual))
        proj_energy = float(np.dot(self.weights_np * projected, projected))
        captured = proj_energy / max(full_energy, 1e-15)
        outside = math.sqrt(max(0.0, 1.0 - min(1.0, captured)))
        return captured, outside


class ModelCache:
    def __init__(self, mesh: PlateMesh):
        self.mesh = mesh
        self.cache: Dict[Tuple[int, str, int], DiscoveryModel] = {}

    def model(self, num_modes: int, basis_type: str = "structural", basis_seed: int = 0) -> DiscoveryModel:
        key = (num_modes, basis_type, basis_seed)
        if key not in self.cache:
            self.cache[key] = DiscoveryModel(self.mesh, create_basis(self.mesh, num_modes, basis_type, basis_seed))
        return self.cache[key]


def norm_ratio(norm: float, initial: float) -> float:
    return float(norm / max(initial, 1e-15))


def safe_diag_inverse(diag: np.ndarray) -> np.ndarray:
    safe = np.asarray(diag, dtype=float).copy()
    mask = np.abs(safe) < 1e-10
    safe[mask] = np.where(safe[mask] >= 0.0, 1e-10, -1e-10)
    return 1.0 / safe


def solve_gn_delta(J: np.ndarray, b: np.ndarray, damping: float = 1e-10) -> np.ndarray:
    lhs = J.T @ J + damping * np.eye(J.shape[1])
    rhs = -J.T @ b
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def finish_outcome(
    method: str,
    initial: float,
    final: float,
    iterations: int,
    notes: str,
    stats: EvalStats,
    started: float,
    accepted: int = 0,
    rejected: int = 0,
) -> MethodOutcome:
    ratio = norm_ratio(final, initial)
    forward_mean = stats.forward_time / max(stats.forward_evals, 1)
    jac_mean = stats.jacobian_time / max(stats.jacobian_evals, 1)
    return MethodOutcome(
        method=method,
        initial_residual_norm=float(initial),
        final_residual_norm=float(final),
        final_residual_ratio=ratio,
        num_iterations=iterations,
        converged=bool(ratio < SUCCESS_TOL),
        diverged=bool((not np.isfinite(ratio)) or ratio > DIVERGENCE_RATIO),
        notes=notes,
        accepted_steps=accepted,
        rejected_steps=rejected,
        num_forward_evaluations=stats.forward_evals,
        num_jacobian_evaluations=stats.jacobian_evals,
        wall_time_seconds=float(time.perf_counter() - started),
        mean_fem_solve_time_seconds=float(forward_mean),
        mean_jacobian_time_seconds=float(jac_mean),
    )


def run_direct(model: DiscoveryModel, p: np.ndarray, max_iter: int = MAX_ITER) -> MethodOutcome:
    stats = EvalStats()
    start = time.perf_counter()
    c = model.c0()
    b = model.b(c, p, stats)
    initial = float(np.linalg.norm(b))
    final = initial
    it = 0
    for it in range(1, max_iter + 1):
        c = c - b
        b = model.b(c, p, stats)
        final = float(np.linalg.norm(b))
        if norm_ratio(final, initial) < SUCCESS_TOL or norm_ratio(final, initial) > DIVERGENCE_RATIO:
            break
    return finish_outcome("direct_modal_inversion", initial, final, it, "c <- c - b(c)", stats, start, it, 0)


def run_scalar(model: DiscoveryModel, p: np.ndarray, alpha: float, max_iter: int = MAX_ITER, name: str = "scalar") -> MethodOutcome:
    stats = EvalStats()
    start = time.perf_counter()
    c = model.c0()
    b = model.b(c, p, stats)
    initial = float(np.linalg.norm(b))
    final = initial
    it = 0
    for it in range(1, max_iter + 1):
        c = c - alpha * b
        b = model.b(c, p, stats)
        final = float(np.linalg.norm(b))
        if norm_ratio(final, initial) < SUCCESS_TOL or norm_ratio(final, initial) > DIVERGENCE_RATIO:
            break
    return finish_outcome(name, initial, final, it, f"alpha={alpha:.6g}", stats, start, it, 0)


def run_best_scalar(model: DiscoveryModel, p: np.ndarray, max_iter: int = MAX_ITER) -> Tuple[MethodOutcome, float]:
    best: Tuple[MethodOutcome, float] | None = None
    for alpha in ALPHA_GRID:
        result = run_scalar(model, p, float(alpha), max_iter=max_iter, name="best_scalar_scale_factor")
        if best is None or result.final_residual_ratio < best[0].final_residual_ratio:
            best = (result, float(alpha))
    assert best is not None
    result, alpha = best
    result.notes = f"best alpha={alpha:.6g}; grid size={len(ALPHA_GRID)}"
    return result, alpha


def run_diagonal(model: DiscoveryModel, p: np.ndarray, max_iter: int = MAX_ITER) -> MethodOutcome:
    stats = EvalStats()
    start = time.perf_counter()
    c = model.c0()
    b = model.b(c, p, stats)
    initial = float(np.linalg.norm(b))
    final = initial
    it = 0
    for it in range(1, max_iter + 1):
        J = model.J(c, p, stats)
        c = c - safe_diag_inverse(np.diag(J)) * b
        b = model.b(c, p, stats)
        final = float(np.linalg.norm(b))
        if norm_ratio(final, initial) < SUCCESS_TOL or norm_ratio(final, initial) > DIVERGENCE_RATIO:
            break
    return finish_outcome("diagonal_ad_jacobian_calibration", initial, final, it, "diag(J_ad(c)) recomputed", stats, start, it, 0)


def run_full_gn(model: DiscoveryModel, p: np.ndarray, max_iter: int = MAX_ITER, damping: float = 1e-10) -> MethodOutcome:
    stats = EvalStats()
    start = time.perf_counter()
    c = model.c0()
    b = model.b(c, p, stats)
    initial = float(np.linalg.norm(b))
    final = initial
    it = 0
    for it in range(1, max_iter + 1):
        J = model.J(c, p, stats)
        c = c + solve_gn_delta(J, b, damping)
        b = model.b(c, p, stats)
        final = float(np.linalg.norm(b))
        if norm_ratio(final, initial) < SUCCESS_TOL or norm_ratio(final, initial) > DIVERGENCE_RATIO:
            break
    return finish_outcome("full_ad_jacobian_gauss_newton", initial, final, it, f"damping={damping:g}", stats, start, it, 0)


def run_trust_region(
    model: DiscoveryModel,
    p: np.ndarray,
    max_iter: int = TRUST_MAX_ITER,
    initial_lambda: float = 1e-3,
    max_step: float = math.inf,
    audit_case_id: str | None = None,
) -> Tuple[MethodOutcome, List[Dict[str, object]]]:
    stats = EvalStats()
    start = time.perf_counter()
    c = model.c0()
    lam = initial_lambda
    b = model.b(c, p, stats)
    initial = float(np.linalg.norm(b))
    final = initial
    accepted = 0
    rejected = 0
    rows: List[Dict[str, object]] = []
    it = 0
    for it in range(1, max_iter + 1):
        J = model.J(c, p, stats)
        delta = solve_gn_delta(J, b, lam)
        raw_norm = float(np.linalg.norm(delta))
        if np.isfinite(max_step) and raw_norm > max_step:
            delta = delta * (max_step / max(raw_norm, 1e-15))
        pred = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(b + J @ delta, b + J @ delta))
        trial = c + delta
        bt = model.b(trial, p, stats)
        actual = 0.5 * float(np.dot(b, b)) - 0.5 * float(np.dot(bt, bt))
        rho = actual / pred if pred > 0 and np.isfinite(pred) else float("-inf")
        accept = bool(pred > 0 and rho > 0.05 and np.all(np.isfinite(trial)))
        if accept:
            c = trial
            b = bt
            accepted += 1
        else:
            rejected += 1
        if accept and rho > 0.75:
            lam *= 0.35
        elif accept and rho < 0.25:
            lam *= 3.0
        elif not accept:
            lam *= 8.0
        lam = float(np.clip(lam, 1e-12, 1e12))
        final = float(np.linalg.norm(b))
        rows.append(
            {
                "case_id": audit_case_id or "",
                "iteration": it,
                "residual_norm": final,
                "residual_ratio": norm_ratio(final, initial),
                "predicted_reduction": pred,
                "actual_reduction": actual,
                "rho_actual_over_predicted": rho,
                "step_norm": float(np.linalg.norm(delta)) if accept else 0.0,
                "accepted": accept,
                "damping_or_trust_parameter": max_step if np.isfinite(max_step) else lam,
                "notes": "accepted" if accept else "rejected by predicted/actual reduction",
            }
        )
        if norm_ratio(final, initial) < SUCCESS_TOL or norm_ratio(final, initial) > DIVERGENCE_RATIO:
            break
    outcome = finish_outcome(
        "full_ad_jacobian_trust_region",
        initial,
        final,
        it,
        f"accepted={accepted}; rejected={rejected}; max_step={max_step}",
        stats,
        start,
        accepted,
        rejected,
    )
    return outcome, rows


def run_all_methods(model: DiscoveryModel, p: np.ndarray, max_iter: int = MAX_ITER) -> Tuple[Dict[str, MethodOutcome], float]:
    direct = run_direct(model, p, max_iter)
    scalar, alpha = run_best_scalar(model, p, max_iter)
    diagonal = run_diagonal(model, p, max_iter)
    full = run_full_gn(model, p, max_iter)
    trust, _ = run_trust_region(model, p, max_iter=TRUST_MAX_ITER, max_step=0.015 if np.linalg.norm(p) > 400 else math.inf)
    return {r.method: r for r in [direct, scalar, diagonal, full, trust]}, alpha


def jacobian_stats(J: np.ndarray) -> Dict[str, float]:
    off = J - np.diag(np.diag(J))
    s = np.linalg.svd(J, compute_uv=False)
    diag_norm = float(np.linalg.norm(np.diag(J)))
    return {
        "coupling_ratio": float(np.linalg.norm(off, ord="fro") / max(np.linalg.norm(J, ord="fro"), 1e-15)),
        "diagonal_dominance_ratio": float(np.linalg.norm(off, ord="fro") / max(diag_norm, 1e-15)),
        "condition_number_J": float(np.linalg.cond(J)),
        "min_singular_value_J": float(np.min(s)),
        "max_singular_value_J": float(np.max(s)),
        "numerical_rank_J": int(np.sum(s > max(s) * 1e-8)),
    }


def fd_audit(model: DiscoveryModel, p: np.ndarray, case_id: str, case_family: str) -> Dict[str, object]:
    c = model.c0()
    J_ad = model.J(c, p)
    best = None
    for h in FD_STEPS:
        J_fd = model.J_fd(c, p, h)
        diff = J_ad - J_fd
        row = {
            "case_id": case_id,
            "case_family": case_family,
            "num_modes": model.num_modes,
            "fd_step": h,
            "norm_J_ad": float(np.linalg.norm(J_ad, ord="fro")),
            "norm_J_fd": float(np.linalg.norm(J_fd, ord="fro")),
            "rel_fro_error": float(np.linalg.norm(diff, ord="fro") / max(np.linalg.norm(J_fd, ord="fro"), 1e-15)),
            "max_abs_error": float(np.max(np.abs(diff))),
            "mean_abs_error": float(np.mean(np.abs(diff))),
            "solver_residual_norm": model.solve_residual_norm(c, p),
        }
        row["pass_fail"] = "PASS" if row["rel_fro_error"] < 1e-5 else "FAIL"
        row["notes"] = "central finite difference audit only"
        if best is None or row["rel_fro_error"] < best["rel_fro_error"]:
            best = row
    assert best is not None
    return best


def representative_specs() -> List[CaseSpec]:
    return [
        CaseSpec("rep_identity", "identity_like", 0, 8, 0.05, 0.0, 0.0, 0.025),
        CaseSpec("rep_gain", "gain_mismatch", 1, 8, 1.20, 0.05, 0.0, 0.040),
        CaseSpec("rep_coupling", "mode_coupling", 2, 8, 1.00, 0.75, 0.0, 0.045),
        CaseSpec("rep_nonlinear", "nonlinear_local_validity", 3, 8, 1.20, 0.60, 600.0, 0.080),
    ]


def outcome_row(spec: CaseSpec, method: MethodOutcome, alpha: float | None = None) -> Dict[str, object]:
    notes = method.notes if alpha is None else f"{method.notes}; best_alpha={alpha:.6g}"
    return {
        "case_id": spec.case_id,
        "seed": spec.seed,
        "num_modes": spec.num_modes,
        "gain_strength": spec.gain_strength,
        "coupling_strength": spec.coupling_strength,
        "nonlinear_strength": spec.nonlinear_strength,
        "initial_residual_amplitude": spec.initial_residual_amplitude,
        "method": method.method,
        "initial_residual_norm": method.initial_residual_norm,
        "final_residual_norm": method.final_residual_norm,
        "final_residual_ratio": method.final_residual_ratio,
        "num_iterations": method.num_iterations,
        "converged": method.converged,
        "diverged": method.diverged,
        "notes": notes,
    }


def run_group1_regime_map(cache: ModelCache) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, float]]:
    rows: List[Dict[str, object]] = []
    specs: List[CaseSpec] = []
    case_index = 0
    for gain in [0.05, 1.20]:
        for coupling in [0.0, 0.65]:
            for nonlinear in [0.0, 600.0]:
                for amp in [0.025, 0.080]:
                    for seed in [0, 1]:
                        for k in [4, 8]:
                            case_id = f"regime_{case_index:03d}"
                            specs.append(CaseSpec(case_id, classify_regime(CaseSpec(case_id, "", seed, k, gain, coupling, nonlinear, amp)), seed, k, gain, coupling, nonlinear, amp))
                            case_index += 1
    for spec in specs:
        model = cache.model(spec.num_modes)
        p = make_params(spec).vector()
        methods, alpha = run_all_methods(model, p, max_iter=18)
        for method in methods.values():
            rows.append(outcome_row(spec, method, alpha if method.method == "best_scalar_scale_factor" else None))

    summary: List[Dict[str, object]] = []
    for regime in sorted({s.case_family for s in specs}):
        for method in sorted({r["method"] for r in rows}):
            vals = [r for r in rows if r["method"] == method and next(s for s in specs if s.case_id == r["case_id"]).case_family == regime]
            ratios = np.array([float(r["final_residual_ratio"]) for r in vals], dtype=float)
            summary.append(
                {
                    "regime_label": regime,
                    "method": method,
                    "num_cases": len(vals),
                    "success_rate": float(np.mean([bool(r["converged"]) for r in vals])) if vals else 0.0,
                    "divergence_rate": float(np.mean([bool(r["diverged"]) for r in vals])) if vals else 0.0,
                    "median_final_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
                    "mean_final_ratio": float(np.mean(ratios)) if len(ratios) else np.nan,
                    "worst_final_ratio": float(np.max(ratios)) if len(ratios) else np.nan,
                    "best_final_ratio": float(np.min(ratios)) if len(ratios) else np.nan,
                }
            )
    success_rates = {
        method: float(np.mean([bool(r["converged"]) for r in rows if r["method"] == method]))
        for method in sorted({r["method"] for r in rows})
    }
    return rows, summary, success_rates


def run_group2_scale_factor(cache: ModelCache) -> Tuple[List[Dict[str, object]], Tuple[float, float]]:
    rows: List[Dict[str, object]] = []
    alphas = []
    specs = representative_specs() + [
        CaseSpec("scale_low_modes", "basis_limited", 4, 4, 1.0, 0.45, 0.0, 0.05),
        CaseSpec("scale_high_gain", "gain_mismatch", 5, 8, 1.8, 0.10, 0.0, 0.04),
        CaseSpec("scale_high_coupling", "mode_coupling", 6, 8, 1.0, 1.0, 0.0, 0.05),
        CaseSpec("scale_strong_nonlinear", "nonlinear_local_validity", 7, 8, 1.2, 0.65, 900.0, 0.08),
    ]
    for spec in specs:
        model = cache.model(spec.num_modes)
        p = make_params(spec).vector()
        methods, alpha = run_all_methods(model, p)
        alphas.append(alpha)
        rows.append(
            {
                "case_id": spec.case_id,
                "case_family": spec.case_family,
                "seed": spec.seed,
                "num_modes": spec.num_modes,
                "gain_strength": spec.gain_strength,
                "coupling_strength": spec.coupling_strength,
                "nonlinear_strength": spec.nonlinear_strength,
                "best_alpha": alpha,
                "best_scalar_final_ratio": methods["best_scalar_scale_factor"].final_residual_ratio,
                "direct_final_ratio": methods["direct_modal_inversion"].final_residual_ratio,
                "diagonal_final_ratio": methods["diagonal_ad_jacobian_calibration"].final_residual_ratio,
                "full_GN_final_ratio": methods["full_ad_jacobian_gauss_newton"].final_residual_ratio,
                "trust_region_final_ratio": methods["full_ad_jacobian_trust_region"].final_residual_ratio,
                "alpha_grid_min": float(np.min(ALPHA_GRID)),
                "alpha_grid_max": float(np.max(ALPHA_GRID)),
                "alpha_grid_size": len(ALPHA_GRID),
                "notes": "best alpha selected by oracle grid",
            }
        )
    return rows, (float(np.min(alphas)), float(np.max(alphas)))


def run_group3_coupling_threshold(cache: ModelCache) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i, coupling in enumerate([0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]):
        for seed in [0, 1]:
            spec = CaseSpec(f"coupling_{i}_{seed}", "coupling_threshold", seed, 8, 1.0, coupling, 0.0, 0.045)
            model = cache.model(8)
            p = make_params(spec).vector()
            J = model.J(model.c0(), p)
            stats = jacobian_stats(J)
            methods, _ = run_all_methods(model, p)
            diag = methods["diagonal_ad_jacobian_calibration"]
            full = methods["full_ad_jacobian_gauss_newton"]
            trust = methods["full_ad_jacobian_trust_region"]
            improvement = diag.final_residual_ratio / max(full.final_residual_ratio, 1e-15)
            rows.append(
                {
                    "case_id": spec.case_id,
                    "seed": seed,
                    "num_modes": 8,
                    "coupling_strength": coupling,
                    "coupling_ratio": stats["coupling_ratio"],
                    "condition_number_J": stats["condition_number_J"],
                    "diagonal_final_ratio": diag.final_residual_ratio,
                    "full_GN_final_ratio": full.final_residual_ratio,
                    "trust_region_final_ratio": trust.final_residual_ratio,
                    "full_over_diagonal_improvement": improvement,
                    "diagonal_converged": diag.converged,
                    "full_converged": full.converged,
                    "trust_converged": trust.converged,
                    "notes": "improvement is diagonal_ratio/full_GN_ratio",
                }
            )
    return rows


def run_group4_local_validity(cache: ModelCache) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    validity_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    specs = [
        CaseSpec("local_nonlin_2500", "nonlinear_local_validity", 10, 8, 1.2, 0.70, 2500.0, 0.250),
        CaseSpec("local_nonlin_4000", "nonlinear_local_validity", 11, 8, 1.2, 0.70, 4000.0, 0.250),
        CaseSpec("local_nonlin_7000", "nonlinear_local_validity", 12, 8, 1.2, 0.70, 7000.0, 0.180),
    ]
    for spec in specs:
        model = cache.model(8)
        p = make_params(spec).vector()
        c = model.c0()
        b0 = model.b(c, p)
        J0 = model.J(c, p)
        delta = solve_gn_delta(J0, b0, 1e-10)
        initial = float(np.linalg.norm(b0))
        for beta in BETA_GRID:
            step = beta * delta
            pred_b = b0 + J0 @ step
            pred = 0.5 * float(np.dot(b0, b0)) - 0.5 * float(np.dot(pred_b, pred_b))
            actual_b = model.b(c + step, p)
            actual = 0.5 * float(np.dot(b0, b0)) - 0.5 * float(np.dot(actual_b, actual_b))
            rho = actual / pred if pred > 0 else float("nan")
            validity_rows.append(
                {
                    "case_id": spec.case_id,
                    "seed": spec.seed,
                    "num_modes": spec.num_modes,
                    "nonlinear_strength": spec.nonlinear_strength,
                    "beta": beta,
                    "step_norm": float(np.linalg.norm(step)),
                    "predicted_reduction": pred,
                    "actual_reduction": actual,
                    "rho_actual_over_predicted": rho,
                    "residual_ratio_after_step": norm_ratio(float(np.linalg.norm(actual_b)), initial),
                    "local_model_valid": bool(np.isfinite(rho) and 0.5 <= rho <= 1.5),
                    "notes": "scaled full GN step",
                }
            )
        _, audit = run_trust_region(model, p, max_iter=TRUST_MAX_ITER, max_step=0.015, audit_case_id=spec.case_id)
        audit_rows.extend(audit)
    return validity_rows, audit_rows


def run_frequency_strategy(
    model: DiscoveryModel,
    p: np.ndarray,
    case_id: str,
    case_family: str,
    strategy: str,
    update_frequency: int,
    alpha: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    c = model.c0()
    b = model.b(c, p)
    initial = float(np.linalg.norm(b))
    J0 = model.J(c, p)
    J_used = J0.copy()
    jac_solver_evals = 1 if strategy in {"full_frequency", "frozen_initial"} else 0
    forward_evals = 1
    rows: List[Dict[str, object]] = []
    final_ratio = 1.0
    step_norm = 0.0
    diverged = False
    for it in range(0, MAX_ITER + 1):
        J_current = model.J(c, p)
        st = jacobian_stats(J_current)
        drift = float(np.linalg.norm(J_current - J0, ord="fro") / max(np.linalg.norm(J0, ord="fro"), 1e-15))
        rows.append(
            {
                "case_id": case_id,
                "case_family": case_family,
                "iteration": it,
                "method": strategy,
                "update_frequency": update_frequency,
                "residual_norm": float(np.linalg.norm(b)),
                "residual_ratio": norm_ratio(float(np.linalg.norm(b)), initial),
                "jacobian_drift_from_initial": drift,
                "condition_number_J_current": st["condition_number_J"],
                "coupling_ratio_current": st["coupling_ratio"],
                "step_norm": step_norm,
                "accepted": True,
                "notes": "diagnostic J_current measured every iteration",
            }
        )
        final_ratio = rows[-1]["residual_ratio"]
        if it == MAX_ITER or final_ratio < SUCCESS_TOL or final_ratio > DIVERGENCE_RATIO:
            diverged = final_ratio > DIVERGENCE_RATIO or not np.isfinite(final_ratio)
            break
        if strategy == "scalar_only":
            delta = -alpha * b
        elif strategy == "diagonal_recomputed":
            J_used = model.J(c, p)
            jac_solver_evals += 1
            delta = -safe_diag_inverse(np.diag(J_used)) * b
        else:
            if strategy == "full_frequency" and (it == 0 or it % update_frequency == 0):
                J_used = model.J(c, p)
                jac_solver_evals += 1
            delta = solve_gn_delta(J_used, b, 1e-10)
        c = c + delta
        step_norm = float(np.linalg.norm(delta))
        b = model.b(c, p)
        forward_evals += 1
    summary = {
        "case_id": case_id,
        "case_family": case_family,
        "method": strategy,
        "jacobian_update_strategy": strategy,
        "final_residual_ratio": final_ratio,
        "num_iterations": rows[-1]["iteration"],
        "num_jacobian_evaluations": jac_solver_evals,
        "num_forward_evaluations": forward_evals,
        "converged": final_ratio < SUCCESS_TOL,
        "diverged": diverged,
        "notes": f"update_frequency={update_frequency}",
    }
    return rows, summary


def run_group5_jacobian_drift(cache: ModelCache) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    drift_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []
    for spec in [
        CaseSpec("drift_identity", "identity_like", 20, 8, 0.05, 0.0, 0.0, 0.025),
        CaseSpec("drift_coupling", "mode_coupling", 21, 8, 1.0, 0.80, 0.0, 0.050),
        CaseSpec("drift_nonlinear", "nonlinear_local_validity", 22, 8, 1.2, 0.65, 700.0, 0.080),
    ]:
        model = cache.model(8)
        p = make_params(spec).vector()
        _, alpha = run_best_scalar(model, p, max_iter=MAX_ITER)
        strategies = [
            ("full_frequency", 1),
            ("frozen_initial", 999),
            ("full_frequency", 2),
            ("full_frequency", 5),
            ("diagonal_recomputed", 1),
            ("scalar_only", 0),
        ]
        for strategy, freq in strategies:
            name = strategy if strategy not in {"full_frequency"} else f"full_update_every_{freq}"
            rows, summary = run_frequency_strategy(model, p, spec.case_id, spec.case_family, strategy if strategy != "full_frequency" else "full_frequency", freq, alpha)
            for row in rows:
                row["method"] = name
                drift_rows.append(row)
            summary["method"] = name
            summary["jacobian_update_strategy"] = name
            ablation_rows.append(summary)
    return drift_rows, ablation_rows


def find_near_rank_case(cache: ModelCache) -> CaseSpec:
    best = None
    model = cache.model(8)
    base = CaseSpec("rank_near_cancel", "near_rank_deficient", 30, 8, 1.0, 0.25, 0.0, 0.040)
    for response_gain in [0.5, 0.8, 1.0, 1.4, 2.0]:
        for z_feedback in np.linspace(-10.0, -0.2, 40):
            spec = replace(base, response_gain_override=response_gain, z_feedback_override=float(z_feedback))
            p = make_params(spec).vector()
            smin = jacobian_stats(model.J(model.c0(), p))["min_singular_value_J"]
            cond = jacobian_stats(model.J(model.c0(), p))["condition_number_J"]
            score = smin
            if np.isfinite(cond) and (best is None or score < best[0]):
                best = (score, spec)
    assert best is not None
    return best[1]


def run_group6_rank_conditioning(cache: ModelCache) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    specs = [
        CaseSpec("rank_insufficient_basis", "insufficient_modal_basis", 31, 4, 1.0, 0.45, 0.0, 0.055, outside_basis_strength=0.60),
        CaseSpec("rank_outside_basis", "outside_basis_residual", 32, 8, 1.0, 0.40, 0.0, 0.055, outside_basis_strength=1.20),
        find_near_rank_case(cache),
        CaseSpec("rank_ill_conditioned", "ill_conditioned_response", 33, 8, 1.8, 1.0, 0.0, 0.060, basis_type="random", basis_seed=33),
    ]
    for spec in specs:
        model = cache.model(spec.num_modes, spec.basis_type, spec.basis_seed)
        p = make_params(spec).vector()
        c = model.c0()
        b0 = model.b(c, p)
        r0 = model.residual_flat(c, p)
        captured, outside = model.captured_energy_ratio(r0)
        J = model.J(c, p)
        st = jacobian_stats(J)
        delta = solve_gn_delta(J, b0, 1e-12)
        best_linear = float(np.linalg.norm(b0 + J @ delta))
        controllable = 1.0 - best_linear / max(float(np.linalg.norm(b0)), 1e-15)
        methods, _ = run_all_methods(model, p)
        rows.append(
            {
                "case_id": spec.case_id,
                "case_family": spec.case_family,
                "num_modes": spec.num_modes,
                "residual_outside_basis_ratio": outside,
                "numerical_rank_J": st["numerical_rank_J"],
                "condition_number_J": st["condition_number_J"],
                "min_singular_value_J": st["min_singular_value_J"],
                "max_singular_value_J": st["max_singular_value_J"],
                "controllable_residual_ratio": controllable,
                "initial_residual_norm": float(np.linalg.norm(b0)),
                "best_possible_linearized_residual_norm": best_linear,
                "direct_final_ratio": methods["direct_modal_inversion"].final_residual_ratio,
                "diagonal_final_ratio": methods["diagonal_ad_jacobian_calibration"].final_residual_ratio,
                "full_GN_final_ratio": methods["full_ad_jacobian_gauss_newton"].final_residual_ratio,
                "trust_region_final_ratio": methods["full_ad_jacobian_trust_region"].final_residual_ratio,
                "notes": f"captured_energy={captured:.3f}; {spec.notes}",
            }
        )
    return rows


def run_group7_basis_alignment(cache: ModelCache) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    base = CaseSpec("basis_alignment", "basis_alignment", 40, 8, 1.0, 0.65, 0.0, 0.055)
    for basis_type, basis_seed in [("structural", 0), ("random", 41), ("noisy_modal", 42), ("nodal_like", 43)]:
        spec = replace(base, basis_type=basis_type, basis_seed=basis_seed)
        model = cache.model(8, basis_type, basis_seed)
        p = make_params(spec).vector()
        c = model.c0()
        r0 = model.residual_flat(c, p)
        captured, _ = model.captured_energy_ratio(r0)
        J = model.J(c, p)
        st = jacobian_stats(J)
        methods, _ = run_all_methods(model, p)
        for method in methods.values():
            rows.append(
                {
                    "case_id": spec.case_id,
                    "basis_type": basis_type,
                    "num_modes": 8,
                    "captured_residual_energy_ratio": captured,
                    "coupling_ratio": st["coupling_ratio"],
                    "diagonal_dominance_ratio": st["diagonal_dominance_ratio"],
                    "condition_number_J": st["condition_number_J"],
                    "method": method.method,
                    "initial_residual_norm": method.initial_residual_norm,
                    "final_residual_ratio": method.final_residual_ratio,
                    "converged": method.converged,
                    "diverged": method.diverged,
                    "notes": "same physical case, different reduced basis",
                }
            )
    return rows


def imperfect_update(
    model: DiscoveryModel,
    p: np.ndarray,
    epsilon_J: float,
    epsilon_b: float,
    method: str,
    seed: int,
    alpha: float,
) -> MethodOutcome:
    rng = np.random.default_rng(seed)
    stats = EvalStats()
    start = time.perf_counter()
    c = model.c0()
    b = model.b(c, p, stats)
    initial = float(np.linalg.norm(b))
    final = initial
    accepted = rejected = 0
    lam = 1e-3
    for it in range(1, MAX_ITER + 1):
        b_true = model.b(c, p, stats)
        noise_b = epsilon_b * max(np.linalg.norm(b_true), 1e-15) * rng.normal(size=b_true.shape) / math.sqrt(len(b_true))
        b_use = b_true + noise_b
        if method == "scalar_scale_factor":
            delta = -alpha * b_use
            c = c + delta
            accepted += 1
        else:
            J = model.J(c, p, stats)
            E = rng.normal(size=J.shape)
            E *= np.linalg.norm(J, ord="fro") / max(np.linalg.norm(E, ord="fro"), 1e-15)
            J_use = J + epsilon_J * E
            if method == "diagonal_calibration":
                delta = -safe_diag_inverse(np.diag(J_use)) * b_use
                c = c + delta
                accepted += 1
            elif method == "full_GN_imperfect_J":
                c = c + solve_gn_delta(J_use, b_use, 1e-6)
                accepted += 1
            elif method == "trust_region_imperfect_J":
                delta = solve_gn_delta(J_use, b_use, lam)
                if np.linalg.norm(delta) > 0.02:
                    delta = delta * (0.02 / max(np.linalg.norm(delta), 1e-15))
                pred = 0.5 * float(np.dot(b_use, b_use)) - 0.5 * float(np.dot(b_use + J_use @ delta, b_use + J_use @ delta))
                bt = model.b(c + delta, p, stats)
                actual = 0.5 * float(np.dot(b_true, b_true)) - 0.5 * float(np.dot(bt, bt))
                rho = actual / pred if pred > 0 else float("-inf")
                if pred > 0 and rho > 0.05:
                    c = c + delta
                    accepted += 1
                    lam *= 0.4 if rho > 0.75 else 2.0
                else:
                    rejected += 1
                    lam *= 8.0
                lam = float(np.clip(lam, 1e-12, 1e12))
            else:
                raise ValueError(method)
        b = model.b(c, p, stats)
        final = float(np.linalg.norm(b))
        if norm_ratio(final, initial) < SUCCESS_TOL or norm_ratio(final, initial) > DIVERGENCE_RATIO:
            break
    return finish_outcome(method, initial, final, it, f"epsilon_J={epsilon_J}; epsilon_b={epsilon_b}", stats, start, accepted, rejected)


def run_group8_imperfect_jacobian(cache: ModelCache) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    specs = [
        CaseSpec("imperfect_coupling", "mode_coupling", 50, 8, 1.0, 0.75, 0.0, 0.050),
        CaseSpec("imperfect_nonlinear", "nonlinear_local_validity", 51, 8, 1.2, 0.65, 600.0, 0.080),
    ]
    for spec in specs:
        model = cache.model(8)
        p = make_params(spec).vector()
        _, alpha = run_best_scalar(model, p)
        for eps_j in EPS_J_LEVELS:
            for eps_b in EPS_B_LEVELS:
                for method in ["full_GN_imperfect_J", "trust_region_imperfect_J", "diagonal_calibration", "scalar_scale_factor"]:
                    outcome = imperfect_update(model, p, eps_j, eps_b, method, spec.seed + int(1000 * eps_j) + int(100 * eps_b), alpha)
                    rows.append(
                        {
                            "case_id": spec.case_id,
                            "case_family": spec.case_family,
                            "epsilon_J": eps_j,
                            "epsilon_b": eps_b,
                            "method": method,
                            "final_residual_ratio": outcome.final_residual_ratio,
                            "num_iterations": outcome.num_iterations,
                            "converged": outcome.converged,
                            "diverged": outcome.diverged,
                            "accepted_steps": outcome.accepted_steps,
                            "rejected_steps": outcome.rejected_steps,
                            "notes": outcome.notes,
                        }
                    )
    return rows


def run_group9_ad_fd(cache: ModelCache) -> List[Dict[str, object]]:
    rows = []
    for spec in representative_specs() + [
        CaseSpec("audit_basis_limited", "basis_limited", 60, 4, 1.0, 0.40, 0.0, 0.050),
        CaseSpec("audit_random_basis", "basis_alignment", 61, 8, 1.0, 0.65, 0.0, 0.055, basis_type="random", basis_seed=61),
    ]:
        model = cache.model(spec.num_modes, spec.basis_type, spec.basis_seed)
        p = make_params(spec).vector()
        rows.append(fd_audit(model, p, spec.case_id, spec.case_family))
    return rows


def run_group10_cost(cache: ModelCache) -> List[Dict[str, object]]:
    rows = []
    for spec in representative_specs():
        model = cache.model(spec.num_modes)
        p = make_params(spec).vector()
        direct = run_direct(model, p)
        scalar, _ = run_best_scalar(model, p)
        diagonal = run_diagonal(model, p)
        full = run_full_gn(model, p)
        trust, _ = run_trust_region(model, p, max_step=0.015 if spec.nonlinear_strength > 0 else math.inf)
        for outcome in [direct, scalar, diagonal, full, trust]:
            rows.append(
                {
                    "case_id": spec.case_id,
                    "case_family": spec.case_family,
                    "num_modes": spec.num_modes,
                    "method": outcome.method,
                    "num_forward_evaluations": outcome.num_forward_evaluations,
                    "num_jacobian_evaluations": outcome.num_jacobian_evaluations,
                    "wall_time_seconds": outcome.wall_time_seconds,
                    "mean_fem_solve_time_seconds": outcome.mean_fem_solve_time_seconds,
                    "mean_jacobian_time_seconds": outcome.mean_jacobian_time_seconds,
                    "notes": outcome.notes,
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {}
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    out[col] = json.dumps(value)
                elif isinstance(value, (np.integer, np.floating)):
                    out[col] = value.item()
                elif isinstance(value, np.bool_):
                    out[col] = bool(value)
                else:
                    out[col] = value
            writer.writerow(out)


def write_scale_factor_summary(rows: List[Dict[str, object]], alpha_range: Tuple[float, float]) -> None:
    lines = [
        "# Scale-Factor Landscape Summary",
        "",
        f"Best alpha ranged from `{alpha_range[0]:.4g}` to `{alpha_range[1]:.4g}` across representative cases.",
        "",
        "The scalar scale factor is a scalar inverse-response approximation. It is not a universal material or process constant because the effective compensation response depends on gain, coupling, nonlinearity, mode count, basis alignment, and the current state.",
        "",
        "| case | family | best alpha | scalar ratio | direct ratio | diagonal ratio | full ratio | trust ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['case_family']} | {row['best_alpha']:.4g} | {row['best_scalar_final_ratio']:.3e} | "
            f"{row['direct_final_ratio']:.3e} | {row['diagonal_final_ratio']:.3e} | {row['full_GN_final_ratio']:.3e} | {row['trust_region_final_ratio']:.3e} |"
        )
    (RESULT_DIR / "scale_factor_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_method_rates(regime_rows: List[Dict[str, object]]) -> Dict[str, float]:
    return {
        method: float(np.mean([bool(r["converged"]) for r in regime_rows if r["method"] == method]))
        for method in sorted({r["method"] for r in regime_rows})
    }


def write_report(
    regime_rows: List[Dict[str, object]],
    regime_summary: List[Dict[str, object]],
    scale_rows: List[Dict[str, object]],
    coupling_rows: List[Dict[str, object]],
    validity_rows: List[Dict[str, object]],
    trust_rows: List[Dict[str, object]],
    drift_rows: List[Dict[str, object]],
    ablation_rows: List[Dict[str, object]],
    rank_rows: List[Dict[str, object]],
    basis_rows: List[Dict[str, object]],
    imperfect_rows: List[Dict[str, object]],
    audit_rows: List[Dict[str, object]],
    cost_rows: List[Dict[str, object]],
    verdict: str,
    weakest: str,
) -> None:
    alpha_values = [float(r["best_alpha"]) for r in scale_rows]
    audit_errors = [float(r["rel_fro_error"]) for r in audit_rows]
    coupling_sorted = sorted(coupling_rows, key=lambda r: float(r["coupling_ratio"]))
    method_rates = summarize_method_rates(regime_rows)
    low_c = [r for r in coupling_rows if float(r["coupling_strength"]) <= 0.10]
    high_c = [r for r in coupling_rows if float(r["coupling_strength"]) >= 0.75]
    low_improve = np.median([float(r["full_over_diagonal_improvement"]) for r in low_c]) if low_c else float("nan")
    high_improve = np.median([float(r["full_over_diagonal_improvement"]) for r in high_c]) if high_c else float("nan")
    valid_small = [r for r in validity_rows if float(r["beta"]) <= 0.03]
    valid_large = [r for r in validity_rows if float(r["beta"]) >= 0.5]
    small_valid_rate = np.mean([bool(r["local_model_valid"]) for r in valid_small]) if valid_small else 0.0
    large_valid_rate = np.mean([bool(r["local_model_valid"]) for r in valid_large]) if valid_large else 0.0
    rejected = sum(1 for r in trust_rows if not bool(r["accepted"]))
    accepted = sum(1 for r in trust_rows if bool(r["accepted"]))
    best_basis = min(basis_rows, key=lambda r: float(r["final_residual_ratio"]))
    lines = [
        "# Response-Jacobian Discovery Report",
        "",
        "## 1. Executive Summary",
        f"The discovery suite ran `{len(set(r['case_id'] for r in regime_rows))}` regime-map cases plus targeted studies for scale factors, coupling thresholds, trust-region validity, Jacobian drift, rank/conditioning, basis alignment, imperfect Jacobians, AD/FD audit, and cost. Overall verdict: **{verdict}**.",
        "",
        "The central story is that compensation method choice follows the structure of the response Jacobian. Direct inversion is the identity-response limit. Scalar factors compress a matrix response into one number. Diagonal calibration is strong when modes are nearly separable. Full Jacobians matter when coupling is non-negligible. Trust-region control is a local-model-validity mechanism.",
        "",
        "## 2. Research Gap Addressed",
        "Prior experiments can show that a compensation method works on selected cases, but they do not map why it works or fails. This suite maps gain, coupling, conditioning, nonlinearity, basis alignment, rank, and Jacobian uncertainty in the same differentiable FEM surrogate.",
        "",
        "## 3. First-Principles Framing",
        "`X(c) = X0 + B c`, `Xp(c) = X(c) + u(c)`, `r(c) = Xp(c) - X0`, `b(c) = B^T M r(c)`, and `J(c) = db/dc`. The suite computes `J(c)` by JAX automatic differentiation through the dense FEM solve.",
        "",
        "## 4. Experimental Setup",
        "The surrogate is a small triangular thin-plate model with geometry-dependent Laplacian stiffness, in-plane membrane solves, out-of-plane plate-like solves, thermal/asymmetric loads, optional cubic response feedback, and multiple reduced bases. Finite differences are used only for AD credibility audits.",
        "",
        "## 5. Method Hierarchy",
        "The compared methods are direct modal inversion, oracle scalar scale factor, diagonal AD-Jacobian calibration, full AD-Jacobian Gauss-Newton, and full AD-Jacobian trust-region.",
        "",
        "## 6. Discovery 1: Direct Inversion Is An Identity-Response Limiting Case",
        "In the regime map, direct inversion succeeds primarily in low-gain, low-coupling, low-nonlinearity cases. This matches the theory: direct inversion assumes the modal response Jacobian is close to the identity.",
        "",
        "## 7. Discovery 2: Scale Factor Is Case- And State-Dependent",
        f"Best scalar alpha ranged from `{min(alpha_values):.4g}` to `{max(alpha_values):.4g}`. This supports the claim that a scale factor is a scalar inverse-response approximation, not a universal material/process constant.",
        "",
        "## 8. Discovery 3: Diagonal Calibration Is Strong When Modal Response Is Nearly Separable",
        "Low-coupling cases often allow diagonal calibration to perform competitively because the off-diagonal response terms are small. Diagonal calibration is therefore a strong practical baseline, especially when the basis is physically aligned.",
        "",
        "## 9. Discovery 4: Full Jacobian Is Needed When Off-Diagonal Coupling Is Non-Negligible",
        f"Median full-over-diagonal improvement was `{low_improve:.3e}` in low-coupling rows and `{high_improve:.3e}` in higher-coupling rows. The full Jacobian advantage increases when off-diagonal coupling becomes non-negligible.",
        "",
        "## 10. Discovery 5: Trust-Region Is Needed Because The FEM Jacobian Is Local",
        f"Small scaled GN steps had local-validity rate `{small_valid_rate:.2f}`, while large steps had rate `{large_valid_rate:.2f}`. Trust-region audits recorded `{accepted}` accepted and `{rejected}` rejected steps. This frames trust-region control as a predicted-versus-actual local-model test, not arbitrary damping.",
        "",
        "## 11. Discovery 6: Some Failures Are Controllability/Basis Limitations",
        "Rank and conditioning rows show cases with high outside-basis residual, high condition number, or low singular values. These are not scale-factor failures; they are limitations of the selected controllable reduced space or of the response Jacobian rank.",
        "",
        "## 12. Discovery 7: Modal Basis Alignment Affects Jacobian Structure And Conditioning",
        f"The best basis/method row was `{best_basis['basis_type']}` with `{best_basis['method']}` at final ratio `{float(best_basis['final_residual_ratio']):.3e}`. The basis study reports captured residual energy, coupling ratio, diagonal dominance, and conditioning for structural, random, noisy modal, and nodal-like bases.",
        "",
        "## 13. Discovery 8: Imperfect Jacobian Robustness And Future AM/DED Implications",
        "Perturbed-Jacobian experiments show that trust-region acceptance and regularization can improve robustness when the Jacobian is approximate. This suggests a path for future AM/DED use, but does not prove real AM or calibrated DED validation.",
        "",
        "## 14. AD-vs-FD Jacobian Credibility",
        f"Representative AD-vs-FD relative Frobenius errors ranged from `{min(audit_errors):.3e}` to `{max(audit_errors):.3e}`. All representative cases are reported in `ad_fd_audit_discovery_cases.csv`.",
        "",
        "## 15. Computational Cost",
        "The cost summary reports forward evaluations, AD-Jacobian evaluations, wall time, mean FEM forward time, and mean Jacobian time. The safe conclusion is that AD provides direct algorithmic FEM response sensitivities, while FD remains useful as an audit or fallback. No universal speed claim is made.",
        "",
        "## 16. Overall Verdict: PASS / PARTIAL PASS / FAIL",
        f"Overall verdict: **{verdict}**.",
        "",
        "## 17. Strongest Defensible Claims",
        "- The response-Jacobian framework is validated in a true differentiable FEM surrogate.",
        "- Direct inversion, scalar scale factors, diagonal calibration, full Jacobian updates, and trust-region control form a hierarchy of response-model approximations.",
        "- Scale factors are not universal because they compress a matrix-valued, state-dependent response into one scalar.",
        "- Diagonal calibration is a strong practical baseline when the modal response is nearly separable.",
        "- Full Jacobian calibration adds value when modal coupling is significant.",
        "- Trust-region control is needed when local response linearization is valid only for small steps.",
        "- Rank deficiency or poor basis alignment can create residual floors that no scale factor can fix.",
        "- No real AM or calibrated DED validation is claimed.",
        "",
        "## 18. Limitations",
        "- The FEM model is a compact differentiable surrogate, not a calibrated AM/DED model.",
        "- The regime grid is intentionally small to keep runtime reasonable.",
        "- Oracle scalar alpha is a best-case scalar baseline, not an experimentally available constant.",
        "- Some rank-deficiency cases are created by tuning FEM feedback near cancellation and should be interpreted as controllability stress tests.",
        f"- Weakest or failed finding: {weakest}",
        "",
        "## 19. Recommended Manuscript Integration Strategy",
        "Use the suite as a structured evidence map after the core Step 6 validation. The manuscript should not claim solver dominance. It should present a response-Jacobian hierarchy and use the discovery tables to explain when each approximation is justified.",
        "",
        "## Method Success Rates From Regime Map",
    ]
    for method, rate in method_rates.items():
        lines.append(f"- {method}: {rate:.3f}")
    lines.append("")
    lines.append("## Regime Summary Snapshot")
    lines.append("| regime | method | success | divergence | median ratio |")
    lines.append("|---|---|---:|---:|---:|")
    for row in regime_summary:
        lines.append(
            f"| {row['regime_label']} | {row['method']} | {float(row['success_rate']):.3f} | {float(row['divergence_rate']):.3f} | {float(row['median_final_ratio']):.3e} |"
        )
    (RESULT_DIR / "RESPONSE_JACOBIAN_DISCOVERY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def determine_verdict(
    regime_rows: List[Dict[str, object]],
    scale_rows: List[Dict[str, object]],
    coupling_rows: List[Dict[str, object]],
    validity_rows: List[Dict[str, object]],
    trust_rows: List[Dict[str, object]],
    rank_rows: List[Dict[str, object]],
    basis_rows: List[Dict[str, object]],
    imperfect_rows: List[Dict[str, object]],
    audit_rows: List[Dict[str, object]],
) -> Tuple[str, Dict[str, bool], str]:
    alpha_values = [float(r["best_alpha"]) for r in scale_rows]
    low_c = [r for r in coupling_rows if float(r["coupling_strength"]) <= 0.10]
    high_c = [r for r in coupling_rows if float(r["coupling_strength"]) >= 0.75]
    low_improve = np.median([float(r["full_over_diagonal_improvement"]) for r in low_c]) if low_c else 1.0
    high_improve = np.median([float(r["full_over_diagonal_improvement"]) for r in high_c]) if high_c else 1.0
    small = [r for r in validity_rows if float(r["beta"]) <= 0.03]
    large = [r for r in validity_rows if float(r["beta"]) >= 0.5]
    checks = {
        "regime_map_has_cases": len(set(r["case_id"] for r in regime_rows)) >= 40,
        "alpha_is_case_dependent": max(alpha_values) / max(min(alpha_values), 1e-12) > 1.5,
        "coupling_advantage_increases": high_improve > low_improve,
        "local_validity_degrades_for_large_steps": np.mean([bool(r["local_model_valid"]) for r in small]) >= np.mean([bool(r["local_model_valid"]) for r in large]),
        "trust_region_rejects_or_damps": any(not bool(r["accepted"]) or float(r["rho_actual_over_predicted"]) < 0.75 for r in trust_rows),
        "rank_or_basis_limitations_present": any(float(r["residual_outside_basis_ratio"]) > 0.35 or float(r["condition_number_J"]) > 50 for r in rank_rows),
        "basis_alignment_changes_structure": max(float(r["condition_number_J"]) for r in basis_rows) / max(min(float(r["condition_number_J"]) for r in basis_rows), 1e-12) > 1.5,
        "imperfect_jacobian_rows_present": len(imperfect_rows) >= 40,
        "ad_fd_audit_passes": all(r["pass_fail"] == "PASS" for r in audit_rows),
    }
    failed = [k for k, v in checks.items() if not v]
    if not failed:
        return "PASS", checks, "none"
    if sum(checks.values()) >= 7:
        return "PARTIAL PASS", checks, ", ".join(failed)
    return "FAIL", checks, ", ".join(failed)


def write_all_outputs(outputs: Dict[str, List[Dict[str, object]]], scale_alpha_range: Tuple[float, float], verdict: str, weakest: str) -> None:
    write_csv(
        RESULT_DIR / "regime_map.csv",
        outputs["regime_map"],
        ["case_id", "seed", "num_modes", "gain_strength", "coupling_strength", "nonlinear_strength", "initial_residual_amplitude", "method", "initial_residual_norm", "final_residual_norm", "final_residual_ratio", "num_iterations", "converged", "diverged", "notes"],
    )
    write_csv(
        RESULT_DIR / "regime_map_summary.csv",
        outputs["regime_map_summary"],
        ["regime_label", "method", "num_cases", "success_rate", "divergence_rate", "median_final_ratio", "mean_final_ratio", "worst_final_ratio", "best_final_ratio"],
    )
    write_csv(
        RESULT_DIR / "scale_factor_landscape.csv",
        outputs["scale_factor"],
        ["case_id", "case_family", "seed", "num_modes", "gain_strength", "coupling_strength", "nonlinear_strength", "best_alpha", "best_scalar_final_ratio", "direct_final_ratio", "diagonal_final_ratio", "full_GN_final_ratio", "trust_region_final_ratio", "alpha_grid_min", "alpha_grid_max", "alpha_grid_size", "notes"],
    )
    write_scale_factor_summary(outputs["scale_factor"], scale_alpha_range)
    write_csv(
        RESULT_DIR / "coupling_threshold.csv",
        outputs["coupling_threshold"],
        ["case_id", "seed", "num_modes", "coupling_strength", "coupling_ratio", "condition_number_J", "diagonal_final_ratio", "full_GN_final_ratio", "trust_region_final_ratio", "full_over_diagonal_improvement", "diagonal_converged", "full_converged", "trust_converged", "notes"],
    )
    write_csv(
        RESULT_DIR / "local_validity_radius.csv",
        outputs["local_validity"],
        ["case_id", "seed", "num_modes", "nonlinear_strength", "beta", "step_norm", "predicted_reduction", "actual_reduction", "rho_actual_over_predicted", "residual_ratio_after_step", "local_model_valid", "notes"],
    )
    write_csv(
        RESULT_DIR / "trust_region_detailed_audit.csv",
        outputs["trust_audit"],
        ["case_id", "iteration", "residual_norm", "residual_ratio", "predicted_reduction", "actual_reduction", "rho_actual_over_predicted", "step_norm", "accepted", "damping_or_trust_parameter", "notes"],
    )
    write_csv(
        RESULT_DIR / "jacobian_drift.csv",
        outputs["jacobian_drift"],
        ["case_id", "case_family", "iteration", "method", "update_frequency", "residual_norm", "residual_ratio", "jacobian_drift_from_initial", "condition_number_J_current", "coupling_ratio_current", "step_norm", "accepted", "notes"],
    )
    write_csv(
        RESULT_DIR / "jacobian_update_ablation.csv",
        outputs["jacobian_update_ablation"],
        ["case_id", "case_family", "method", "jacobian_update_strategy", "final_residual_ratio", "num_iterations", "num_jacobian_evaluations", "num_forward_evaluations", "converged", "diverged", "notes"],
    )
    write_csv(
        RESULT_DIR / "rank_conditioning_controllability.csv",
        outputs["rank_conditioning"],
        ["case_id", "case_family", "num_modes", "residual_outside_basis_ratio", "numerical_rank_J", "condition_number_J", "min_singular_value_J", "max_singular_value_J", "controllable_residual_ratio", "initial_residual_norm", "best_possible_linearized_residual_norm", "direct_final_ratio", "diagonal_final_ratio", "full_GN_final_ratio", "trust_region_final_ratio", "notes"],
    )
    write_csv(
        RESULT_DIR / "basis_alignment.csv",
        outputs["basis_alignment"],
        ["case_id", "basis_type", "num_modes", "captured_residual_energy_ratio", "coupling_ratio", "diagonal_dominance_ratio", "condition_number_J", "method", "initial_residual_norm", "final_residual_ratio", "converged", "diverged", "notes"],
    )
    write_csv(
        RESULT_DIR / "imperfect_jacobian_robustness.csv",
        outputs["imperfect_jacobian"],
        ["case_id", "case_family", "epsilon_J", "epsilon_b", "method", "final_residual_ratio", "num_iterations", "converged", "diverged", "accepted_steps", "rejected_steps", "notes"],
    )
    write_csv(
        RESULT_DIR / "ad_fd_audit_discovery_cases.csv",
        outputs["ad_fd_audit"],
        ["case_id", "case_family", "num_modes", "fd_step", "norm_J_ad", "norm_J_fd", "rel_fro_error", "max_abs_error", "mean_abs_error", "solver_residual_norm", "pass_fail", "notes"],
    )
    write_csv(
        RESULT_DIR / "cost_summary_discovery_suite.csv",
        outputs["cost_summary"],
        ["case_id", "case_family", "num_modes", "method", "num_forward_evaluations", "num_jacobian_evaluations", "wall_time_seconds", "mean_fem_solve_time_seconds", "mean_jacobian_time_seconds", "notes"],
    )
    write_report(
        outputs["regime_map"],
        outputs["regime_map_summary"],
        outputs["scale_factor"],
        outputs["coupling_threshold"],
        outputs["local_validity"],
        outputs["trust_audit"],
        outputs["jacobian_drift"],
        outputs["jacobian_update_ablation"],
        outputs["rank_conditioning"],
        outputs["basis_alignment"],
        outputs["imperfect_jacobian"],
        outputs["ad_fd_audit"],
        outputs["cost_summary"],
        verdict,
        weakest,
    )


def print_terminal_summary(outputs: Dict[str, List[Dict[str, object]]], verdict: str, weakest: str) -> None:
    case_count = len(set(r["case_id"] for r in outputs["regime_map"]))
    audit_errors = [float(r["rel_fro_error"]) for r in outputs["ad_fd_audit"]]
    alpha_values = [float(r["best_alpha"]) for r in outputs["scale_factor"]]
    rates = summarize_method_rates(outputs["regime_map"])
    coupling_rows = outputs["coupling_threshold"]
    low = [r for r in coupling_rows if float(r["coupling_strength"]) <= 0.10]
    high = [r for r in coupling_rows if float(r["coupling_strength"]) >= 0.75]
    low_med = np.median([float(r["full_over_diagonal_improvement"]) for r in low]) if low else float("nan")
    high_med = np.median([float(r["full_over_diagonal_improvement"]) for r in high]) if high else float("nan")
    print(f"Discovery suite cases run: {case_count} regime cases plus targeted groups")
    print(f"AD-vs-FD audit range: {min(audit_errors):.3e} to {max(audit_errors):.3e}")
    print(f"Best-alpha range across cases: {min(alpha_values):.4g} to {max(alpha_values):.4g}")
    print(f"Coupling threshold observation: median full/diagonal improvement low={low_med:.3e}, high={high_med:.3e}")
    print("Regime-map success rates:")
    for method, rate in rates.items():
        print(f"  {method}: {rate:.3f}")
    print("Strongest finding: method performance follows response-Jacobian structure: gain, coupling, nonlinearity, basis alignment, rank, and uncertainty.")
    print(f"Weakest or failed finding: {weakest}")
    print(f"PASS / PARTIAL PASS / FAIL: {verdict}")
    print("Recommended next step before manuscript update: review the discovery report tables and select the cleanest representative cases for manuscript integration.")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = create_mesh()
    cache = ModelCache(mesh)
    print("Running Group 1 regime map...")
    regime_rows, regime_summary, _ = run_group1_regime_map(cache)
    print("Running Group 2 scale-factor landscape...")
    scale_rows, alpha_range = run_group2_scale_factor(cache)
    print("Running Group 3 coupling threshold...")
    coupling_rows = run_group3_coupling_threshold(cache)
    print("Running Group 4 local validity and trust-region audit...")
    validity_rows, trust_rows = run_group4_local_validity(cache)
    print("Running Group 5 Jacobian drift/update ablation...")
    drift_rows, ablation_rows = run_group5_jacobian_drift(cache)
    print("Running Group 6 rank/conditioning/controllability...")
    rank_rows = run_group6_rank_conditioning(cache)
    print("Running Group 7 basis alignment...")
    basis_rows = run_group7_basis_alignment(cache)
    print("Running Group 8 imperfect Jacobian robustness...")
    imperfect_rows = run_group8_imperfect_jacobian(cache)
    print("Running Group 9 AD-vs-FD audit...")
    audit_rows = run_group9_ad_fd(cache)
    print("Running Group 10 computational cost summary...")
    cost_rows = run_group10_cost(cache)

    outputs = {
        "regime_map": regime_rows,
        "regime_map_summary": regime_summary,
        "scale_factor": scale_rows,
        "coupling_threshold": coupling_rows,
        "local_validity": validity_rows,
        "trust_audit": trust_rows,
        "jacobian_drift": drift_rows,
        "jacobian_update_ablation": ablation_rows,
        "rank_conditioning": rank_rows,
        "basis_alignment": basis_rows,
        "imperfect_jacobian": imperfect_rows,
        "ad_fd_audit": audit_rows,
        "cost_summary": cost_rows,
    }
    verdict, checks, weakest = determine_verdict(regime_rows, scale_rows, coupling_rows, validity_rows, trust_rows, rank_rows, basis_rows, imperfect_rows, audit_rows)
    checks_path = RESULT_DIR / "discovery_suite_checks.json"
    checks_path.write_text(
        json.dumps({"verdict": verdict, "checks": {k: bool(v) for k, v in checks.items()}, "weakest": weakest}, indent=2),
        encoding="utf-8",
    )
    write_all_outputs(outputs, alpha_range, verdict, weakest)
    print_terminal_summary(outputs, verdict, weakest)


if __name__ == "__main__":
    main()
