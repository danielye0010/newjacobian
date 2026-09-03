"""Core geometry, basis, and differentiable response for fixed-bottom FDM study.

The model is a transparent voxel-spring eigenstrain surrogate. It is intended
for response-Jacobian diagnostics, not calibrated FDM process prediction.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from jax import config

config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


MAX_MODES = 32
FD_STEPS = (3e-4, 1e-4, 3e-5, 1e-5, 3e-6)


@dataclass(frozen=True)
class VolumeMesh:
    name: str
    nodes: np.ndarray
    hexes: np.ndarray
    edges: np.ndarray
    bottom_mask: np.ndarray
    surface_mask: np.ndarray
    weights_node: np.ndarray
    weights_dof: np.ndarray
    dimensions: str
    mesh_resolution: str
    notes: str

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_dofs(self) -> int:
        return int(3 * self.num_nodes)

    @property
    def num_elements(self) -> int:
        return int(self.hexes.shape[0])

    @property
    def bottom_nodes(self) -> int:
        return int(np.sum(self.bottom_mask))

    @property
    def surface_nodes(self) -> int:
        return int(np.sum(self.surface_mask))

    @property
    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.nodes.min(axis=0), self.nodes.max(axis=0)


@dataclass(frozen=True)
class ResponseCase:
    name: str
    geometry: str
    purpose: str
    base_strain: float
    gradient_z: float
    lateral_quadratic: float
    twist_xyz: float
    edge_lift: float
    finger_variation: float
    shear_xy: float
    anisotropy: float
    geometry_feedback: float
    nonlinear_feedback: float
    response_gain: float
    stiffness: float
    bending: float
    edge_power: float
    trust_initial_lambda: float
    trust_max_step: float

    def formula_summary(self) -> str:
        return (
            "epsilon = base + gradient_z*z_n + lateral*y_n^2 + twist*x_n*y_n*z_n "
            "+ edge_lift*edge_profile + finger_variation*sin(6*pi*x_01)*edge_profile; "
            "smooth geometry feedback is evaluated on the compensated geometry"
        )


@dataclass(frozen=True)
class BasisData:
    psi: np.ndarray
    eigenvalues: np.ndarray
    descriptions: Tuple[str, ...]
    bottom_error_f: float
    bottom_error_inf: float
    orthonormality_error_f: float
    scalar_condition_estimate: float

    @property
    def usable_modes(self) -> int:
        return int(self.psi.shape[1])


def _structured_hex_mesh(
    name: str,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    active_cell: Callable[[int, int, int], bool],
    dimensions: str,
    notes: str,
) -> VolumeMesh:
    node_map: Dict[Tuple[int, int, int], int] = {}
    nodes: List[Tuple[float, float, float]] = []
    hexes: List[Tuple[int, ...]] = []
    cell_volumes: List[float] = []

    def node(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append((float(xs[i]), float(ys[j]), float(zs[k])))
        return node_map[key]

    for k in range(len(zs) - 1):
        for j in range(len(ys) - 1):
            for i in range(len(xs) - 1):
                if not active_cell(i, j, k):
                    continue
                corners = (
                    node(i, j, k),
                    node(i + 1, j, k),
                    node(i + 1, j + 1, k),
                    node(i, j + 1, k),
                    node(i, j, k + 1),
                    node(i + 1, j, k + 1),
                    node(i + 1, j + 1, k + 1),
                    node(i, j + 1, k + 1),
                )
                hexes.append(corners)
                cell_volumes.append(float((xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j]) * (zs[k + 1] - zs[k])))

    node_array = np.asarray(nodes, dtype=float)
    hex_array = np.asarray(hexes, dtype=int)
    if node_array.size == 0 or hex_array.size == 0:
        raise ValueError(f"{name} geometry is empty")

    edge_set = set()
    face_counts: Dict[Tuple[int, ...], int] = {}
    local_faces = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    for cell in hex_array:
        for a, b in itertools.combinations(cell.tolist(), 2):
            edge_set.add((min(a, b), max(a, b)))
        for face in local_faces:
            key = tuple(sorted(int(cell[q]) for q in face))
            face_counts[key] = face_counts.get(key, 0) + 1

    surface_nodes = set()
    for face, count in face_counts.items():
        if count == 1:
            surface_nodes.update(face)

    weights = np.zeros(node_array.shape[0], dtype=float)
    for cell, volume in zip(hex_array, cell_volumes):
        weights[cell] += volume / 8.0
    positive_mean = float(np.mean(weights[weights > 0]))
    weights[weights <= 0] = positive_mean
    weights /= positive_mean

    zmin = float(np.min(node_array[:, 2]))
    bottom_mask = np.isclose(node_array[:, 2], zmin, atol=1e-12)
    surface_mask = np.zeros(node_array.shape[0], dtype=bool)
    surface_mask[list(surface_nodes)] = True
    return VolumeMesh(
        name=name,
        nodes=node_array,
        hexes=hex_array,
        edges=np.asarray(sorted(edge_set), dtype=int),
        bottom_mask=bottom_mask,
        surface_mask=surface_mask,
        weights_node=weights,
        weights_dof=np.repeat(weights, 3),
        dimensions=dimensions,
        mesh_resolution=f"{len(xs)} x {len(ys)} x {len(zs)} nodes before slot removal",
        notes=notes,
    )


def generate_plate() -> VolumeMesh:
    xs = np.linspace(0.0, 120.0, 14)
    ys = np.linspace(0.0, 30.0, 5)
    zs = np.linspace(0.0, 1.2, 3)
    return _structured_hex_mesh(
        "thin_plate",
        xs,
        ys,
        zs,
        lambda i, j, k: True,
        "120 x 30 x 1.2 mm",
        "structured three-layer thin plate; all nodes on z=0 are fixed only in the compensated input geometry",
    )


def generate_comb() -> VolumeMesh:
    xs = np.linspace(0.0, 120.0, 14)
    ys = np.linspace(0.0, 30.0, 7)
    zs = np.linspace(0.0, 1.2, 3)

    def active(i: int, j: int, k: int) -> bool:
        del k
        base_spine = j < 2
        finger_band = (i % 3) in (0, 1)
        return bool(base_spine or finger_band)

    return _structured_hex_mesh(
        "comb_coupon",
        xs,
        ys,
        zs,
        active,
        "120 x 30 x 1.2 mm bounding box with repeated slots",
        "two-cell base spine with repeated two-cell fingers separated by one-cell slots",
    )


def generate_geometries() -> List[VolumeMesh]:
    return [generate_plate(), generate_comb()]


def response_cases() -> List[ResponseCase]:
    return [
        ResponseCase(
            "plate_pure_bending",
            "thin_plate",
            "clean through-thickness eigenstrain-gradient bending sanity case",
            base_strain=-0.0010,
            gradient_z=-0.0050,
            lateral_quadratic=0.0000,
            twist_xyz=0.0000,
            edge_lift=0.0000,
            finger_variation=0.0000,
            shear_xy=0.0000,
            anisotropy=0.10,
            geometry_feedback=0.04,
            nonlinear_feedback=0.0,
            response_gain=0.55,
            stiffness=1.00,
            bending=0.012,
            edge_power=0.7,
            trust_initial_lambda=1e-4,
            trust_max_step=np.inf,
        ),
        ResponseCase(
            "plate_asymmetric_twist",
            "thin_plate",
            "asymmetric x-y-z eigenstrain and shear producing modal twist coupling",
            base_strain=-0.0012,
            gradient_z=-0.0045,
            lateral_quadratic=-0.0015,
            twist_xyz=0.0055,
            edge_lift=0.0000,
            finger_variation=0.0000,
            shear_xy=0.0040,
            anisotropy=0.28,
            geometry_feedback=0.22,
            nonlinear_feedback=0.0,
            response_gain=0.90,
            stiffness=0.82,
            bending=0.009,
            edge_power=0.9,
            trust_initial_lambda=3e-4,
            trust_max_step=np.inf,
        ),
        ResponseCase(
            "comb_edge_lift",
            "comb_coupon",
            "stronger upper-surface and free-edge shrinkage producing finger-end lift",
            base_strain=-0.0014,
            gradient_z=-0.0040,
            lateral_quadratic=-0.0010,
            twist_xyz=0.0015,
            edge_lift=-0.0065,
            finger_variation=0.0015,
            shear_xy=0.0015,
            anisotropy=0.22,
            geometry_feedback=0.16,
            nonlinear_feedback=0.0,
            response_gain=0.85,
            stiffness=0.78,
            bending=0.008,
            edge_power=0.9,
            trust_initial_lambda=5e-4,
            trust_max_step=np.inf,
        ),
        ResponseCase(
            "comb_finger_varying_shrinkage",
            "comb_coupon",
            "finger-to-finger shrinkage variation with local-global coupling and smooth nonlinear feedback",
            base_strain=-0.0012,
            gradient_z=-0.0035,
            lateral_quadratic=-0.0012,
            twist_xyz=0.0030,
            edge_lift=-0.0045,
            finger_variation=0.0070,
            shear_xy=0.0035,
            anisotropy=0.35,
            geometry_feedback=0.38,
            nonlinear_feedback=0.18,
            response_gain=1.05,
            stiffness=0.66,
            bending=0.006,
            edge_power=1.0,
            trust_initial_lambda=1e-3,
            trust_max_step=0.30,
        ),
    ]


def _reference_laplacian(mesh: VolumeMesh) -> np.ndarray:
    n = mesh.num_nodes
    a = mesh.edges[:, 0]
    b = mesh.edges[:, 1]
    lengths = np.linalg.norm(mesh.nodes[b] - mesh.nodes[a], axis=1)
    w = 1.0 / np.maximum(lengths, 1e-12)
    L = np.zeros((n, n), dtype=float)
    np.add.at(L, (a, a), w)
    np.add.at(L, (b, b), w)
    np.add.at(L, (a, b), -w)
    np.add.at(L, (b, a), -w)
    return L


def compute_fixed_bottom_modes(mesh: VolumeMesh, max_modes: int = MAX_MODES) -> BasisData:
    free = np.where(~mesh.bottom_mask)[0]
    if free.size < 12:
        raise ValueError(f"{mesh.name} has too few non-bottom nodes for a modal basis")

    L = _reference_laplacian(mesh)
    K = L + 0.010 * (L @ L)
    Kff = K[np.ix_(free, free)]
    mass = mesh.weights_node[free]
    inv_sqrt_m = 1.0 / np.sqrt(np.maximum(mass, 1e-14))
    A = inv_sqrt_m[:, None] * Kff * inv_sqrt_m[None, :]
    evals, evecs = np.linalg.eigh(0.5 * (A + A.T))
    positive = np.where(evals > 1e-10)[0]
    if positive.size == 0:
        raise ValueError(f"{mesh.name} constrained mode solve returned no positive modes")

    scalar_count = min(max(16, int(np.ceil(max_modes / 2))), positive.size)
    scalar_ids = positive[:scalar_count]
    scalar_evals = evals[scalar_ids]
    scalar_shapes = inv_sqrt_m[:, None] * evecs[:, scalar_ids]

    direction_specs = (
        (2, 0.055, "out-of-plane bending/twist"),
        (1, 0.82, "in-plane transverse/finger"),
        (0, 1.00, "in-plane longitudinal"),
    )
    candidates: List[Tuple[float, int, int, str]] = []
    for scalar_order, eig in enumerate(scalar_evals):
        for direction, factor, label in direction_specs:
            candidates.append((float(eig * factor), scalar_order, direction, label))
    candidates.sort(key=lambda row: row[0])
    selected = candidates[: min(max_modes, len(candidates))]

    psi = np.zeros((mesh.num_dofs, len(selected)), dtype=float)
    mode_evals = []
    descriptions = []
    for col, (eig, scalar_order, direction, label) in enumerate(selected):
        shape = np.zeros(mesh.num_nodes, dtype=float)
        shape[free] = scalar_shapes[:, scalar_order]
        psi[direction::3, col] = shape
        mode_evals.append(eig)
        descriptions.append(f"{label}, constrained scalar order {scalar_order + 1}")

    gram = psi.T @ (mesh.weights_dof[:, None] * psi)
    bottom_dofs = np.repeat(mesh.bottom_mask, 3)
    bottom_block = psi[bottom_dofs, :]
    bottom_f = float(np.linalg.norm(bottom_block, ord="fro"))
    bottom_inf = float(np.max(np.abs(bottom_block))) if bottom_block.size else 0.0
    orth = float(np.linalg.norm(gram - np.eye(psi.shape[1]), ord="fro"))
    cond = float(mode_evals[-1] / max(mode_evals[0], 1e-15))
    return BasisData(
        psi=psi,
        eigenvalues=np.asarray(mode_evals, dtype=float),
        descriptions=tuple(descriptions),
        bottom_error_f=bottom_f,
        bottom_error_inf=bottom_inf,
        orthonormality_error_f=orth,
        scalar_condition_estimate=cond,
    )


def compute_free_modes(mesh: VolumeMesh, max_modes: int = MAX_MODES) -> BasisData:
    """Compute ordinary unconstrained smooth modes for authority comparisons."""
    L = _reference_laplacian(mesh)
    K = L + 0.010 * (L @ L)
    mass = mesh.weights_node
    inv_sqrt_m = 1.0 / np.sqrt(np.maximum(mass, 1e-14))
    A = inv_sqrt_m[:, None] * K * inv_sqrt_m[None, :]
    evals, evecs = np.linalg.eigh(0.5 * (A + A.T))
    usable = np.where(evals > -1e-10)[0]
    scalar_count = min(max(16, int(np.ceil(max_modes / 2))), usable.size)
    scalar_ids = usable[:scalar_count]
    scalar_evals = np.maximum(evals[scalar_ids], 0.0)
    scalar_shapes = inv_sqrt_m[:, None] * evecs[:, scalar_ids]

    direction_specs = (
        (2, 0.055, "free out-of-plane bending/translation"),
        (1, 0.82, "free in-plane transverse"),
        (0, 1.00, "free in-plane longitudinal"),
    )
    candidates: List[Tuple[float, int, int, str]] = []
    for scalar_order, eig in enumerate(scalar_evals):
        for direction, factor, label in direction_specs:
            candidates.append((float(eig * factor), scalar_order, direction, label))
    candidates.sort(key=lambda row: row[0])
    selected = candidates[: min(max_modes, len(candidates))]

    psi = np.zeros((mesh.num_dofs, len(selected)), dtype=float)
    mode_evals = []
    descriptions = []
    for col, (eig, scalar_order, direction, label) in enumerate(selected):
        psi[direction::3, col] = scalar_shapes[:, scalar_order]
        mode_evals.append(eig)
        descriptions.append(f"{label}, scalar order {scalar_order + 1}")

    gram = psi.T @ (mesh.weights_dof[:, None] * psi)
    bottom_dofs = np.repeat(mesh.bottom_mask, 3)
    bottom_block = psi[bottom_dofs, :]
    bottom_f = float(np.linalg.norm(bottom_block, ord="fro"))
    bottom_inf = float(np.max(np.abs(bottom_block))) if bottom_block.size else 0.0
    orth = float(np.linalg.norm(gram - np.eye(psi.shape[1]), ord="fro"))
    positive_modes = [value for value in mode_evals if value > 1e-12]
    first_positive = min(positive_modes, default=1.0)
    cond = float(max(mode_evals, default=1.0) / max(first_positive, 1e-15))
    return BasisData(
        psi=psi,
        eigenvalues=np.asarray(mode_evals, dtype=float),
        descriptions=tuple(descriptions),
        bottom_error_f=bottom_f,
        bottom_error_inf=bottom_inf,
        orthonormality_error_f=orth,
        scalar_condition_estimate=cond,
    )


def _nearest_node(nodes: np.ndarray, target: Sequence[float]) -> int:
    target_array = np.asarray(target, dtype=float)
    return int(np.argmin(np.linalg.norm(nodes - target_array[None, :], axis=1)))


class FixedBottomJaxResponse:
    """Differentiable released eigenstrain response with bottom-admissible input."""

    def __init__(self, mesh: VolumeMesh, basis: BasisData, case: ResponseCase, mode_count: int):
        if mode_count > basis.usable_modes:
            raise ValueError(f"requested {mode_count} modes but only {basis.usable_modes} are available")
        self.mesh = mesh
        self.case = case
        self.mode_count = int(mode_count)
        self.psi_np = np.asarray(basis.psi[:, :mode_count], dtype=float)
        self.nodes_np = np.asarray(mesh.nodes, dtype=float)
        self.nodes_flat_np = self.nodes_np.reshape(-1)
        self.weights_dof_np = np.asarray(mesh.weights_dof, dtype=float)
        self.bottom_dof_mask_np = np.repeat(mesh.bottom_mask, 3)

        lo, hi = mesh.bbox
        span = np.maximum(hi - lo, 1e-12)
        self.lo_np = lo
        self.span_np = span

        self.nodes = jnp.asarray(self.nodes_np, dtype=jnp.float64)
        self.nodes_flat = jnp.asarray(self.nodes_flat_np, dtype=jnp.float64)
        self.psi = jnp.asarray(self.psi_np, dtype=jnp.float64)
        self.weights = jnp.asarray(self.weights_dof_np, dtype=jnp.float64)
        self.edges = jnp.asarray(mesh.edges, dtype=jnp.int32)
        self.lo = jnp.asarray(lo, dtype=jnp.float64)
        self.span = jnp.asarray(span, dtype=jnp.float64)
        self.ref_lengths = jnp.asarray(
            np.linalg.norm(self.nodes_np[mesh.edges[:, 1]] - self.nodes_np[mesh.edges[:, 0]], axis=1),
            dtype=jnp.float64,
        )

        n = mesh.num_nodes
        xmin, ymin, zmin = lo
        xmax, ymax, _ = hi
        anchor0 = _nearest_node(mesh.nodes, (xmin, ymin, zmin))
        anchor1 = _nearest_node(mesh.nodes, (xmax, ymin, zmin))
        anchor2 = _nearest_node(mesh.nodes, (xmin, ymax, zmin))
        gauge_x = np.zeros(n, dtype=float)
        gauge_y = np.zeros(n, dtype=float)
        gauge_z = np.zeros(n, dtype=float)
        gauge_x[anchor0] = 1.0
        gauge_y[[anchor0, anchor1]] = 1.0
        gauge_z[[anchor0, anchor1, anchor2]] = 1.0
        self.gauge_x = jnp.asarray(gauge_x, dtype=jnp.float64)
        self.gauge_y = jnp.asarray(gauge_y, dtype=jnp.float64)
        self.gauge_z = jnp.asarray(gauge_z, dtype=jnp.float64)

        self._residual_q_jit = jax.jit(self._residual_from_q)
        self._residual_c_jit = jax.jit(self._residual_from_c)
        self._b_jit = jax.jit(self._projected_residual)
        self._J_jit = jax.jit(jax.jacfwd(self._projected_residual))
        self._A_jit = jax.jit(jax.jacfwd(self._residual_from_c))

    def _edge_laplacian(self, X: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        a = self.edges[:, 0]
        b = self.edges[:, 1]
        d = X[b] - X[a]
        lengths = jnp.sqrt(jnp.sum(d * d, axis=1) + 1e-18)
        ratio = lengths / jnp.maximum(self.ref_lengths, 1e-12)
        edge_weights = (1.0 / jnp.maximum(self.ref_lengths, 1e-12)) / (ratio**self.case.edge_power)
        n = self.mesh.num_nodes
        row = jnp.concatenate([a, b, a, b])
        col = jnp.concatenate([a, b, b, a])
        data = jnp.concatenate([edge_weights, edge_weights, -edge_weights, -edge_weights])
        L = jnp.zeros((n, n), dtype=jnp.float64).at[row, col].add(data)
        return L, edge_weights

    def _eigenstrain_targets(self, X: jnp.ndarray) -> jnp.ndarray:
        a = self.edges[:, 0]
        b = self.edges[:, 1]
        d = X[b] - X[a]
        midpoint = 0.5 * (X[a] + X[b])
        midpoint0 = 0.5 * (self.nodes[a] + self.nodes[b])
        normalized = 2.0 * (midpoint - self.lo[None, :]) / self.span[None, :] - 1.0
        normalized0 = 2.0 * (midpoint0 - self.lo[None, :]) / self.span[None, :] - 1.0
        x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2]
        x0, y0, z0 = normalized0[:, 0], normalized0[:, 1], normalized0[:, 2]
        x01 = 0.5 * (x0 + 1.0)
        y01 = 0.5 * (y0 + 1.0)
        z01 = 0.5 * (z0 + 1.0)
        qmid = (midpoint - midpoint0) / self.span[None, :]

        edge_profile = y01**2 * (0.30 + 0.70 * z01)
        finger_profile = jnp.sin(6.0 * jnp.pi * x01) * edge_profile
        eps = (
            self.case.base_strain
            + self.case.gradient_z * z0
            + self.case.lateral_quadratic * y0**2
            + self.case.twist_xyz * x0 * y0 * z0
            + self.case.edge_lift * edge_profile
            + self.case.finger_variation * finger_profile
            + 0.15 * self.case.geometry_feedback * (0.55 * qmid[:, 2] * x0 + 0.25 * qmid[:, 1] * z0)
            + self.case.nonlinear_feedback * (qmid[:, 2] ** 2 + 0.25 * qmid[:, 0] * qmid[:, 1])
        )

        exx = eps * (1.0 + self.case.anisotropy)
        eyy = eps * (1.0 - self.case.anisotropy)
        ezz = 0.18 * eps
        shear = self.case.shear_xy * (0.35 + 0.65 * z01) * (x0 + 0.45 * y0)
        tx = exx * d[:, 0] + shear * d[:, 1] + 0.15 * shear * d[:, 2]
        ty = shear * d[:, 0] + eyy * d[:, 1] - 0.10 * shear * d[:, 2]
        tz = 0.18 * shear * d[:, 0] - 0.12 * shear * d[:, 1] + ezz * d[:, 2]
        qedge = (X[b] - self.nodes[b]) - (X[a] - self.nodes[a])
        feedback = self.case.geometry_feedback * jnp.stack(
            [
                0.30 * qedge[:, 0] + 0.20 * qedge[:, 1] + 0.55 * y0 * qedge[:, 2],
                -0.25 * qedge[:, 0] + 0.45 * qedge[:, 1] - 0.50 * x0 * qedge[:, 2],
                0.65 * y0 * qedge[:, 0]
                - 0.55 * x0 * qedge[:, 1]
                + (0.35 + 0.85 * x0 * y0 + 0.40 * y0) * qedge[:, 2],
            ],
            axis=1,
        )
        nonlinear_scale = self.case.nonlinear_feedback * (
            qmid[:, 2] ** 2 + 0.25 * qmid[:, 0] ** 2 + 0.25 * qmid[:, 1] ** 2
        )
        nonlinear_target = nonlinear_scale[:, None] * jnp.stack(
            [0.35 * d[:, 0] + 0.20 * d[:, 2], -0.25 * d[:, 0] + 0.30 * d[:, 1], d[:, 2] + 0.15 * d[:, 0]],
            axis=1,
        )
        return self.case.response_gain * (jnp.stack([tx, ty, tz], axis=1) + feedback + nonlinear_target)

    def _rhs_from_targets(self, targets: jnp.ndarray, edge_weights: jnp.ndarray) -> jnp.ndarray:
        a = self.edges[:, 0]
        b = self.edges[:, 1]
        weighted = edge_weights[:, None] * targets
        rhs = jnp.zeros((self.mesh.num_nodes, 3), dtype=jnp.float64)
        rhs = rhs.at[a].add(-weighted)
        rhs = rhs.at[b].add(weighted)
        return rhs

    def _solve_release(self, X: jnp.ndarray) -> jnp.ndarray:
        L, edge_weights = self._edge_laplacian(X)
        targets = self._eigenstrain_targets(X)
        rhs = self._rhs_from_targets(targets, edge_weights)
        eye = jnp.eye(self.mesh.num_nodes, dtype=jnp.float64)
        K = self.case.stiffness * L + self.case.bending * (L @ L) + 1e-8 * eye
        gauge_strength = 25.0
        Kx = K + jnp.diag(gauge_strength * self.gauge_x)
        Ky = K + jnp.diag(gauge_strength * self.gauge_y)
        Kz = K + jnp.diag(gauge_strength * self.gauge_z)
        ux = jnp.linalg.solve(Kx, rhs[:, 0])
        uy = jnp.linalg.solve(Ky, rhs[:, 1])
        uz = jnp.linalg.solve(Kz, rhs[:, 2])
        return jnp.stack([ux, uy, uz], axis=1)

    def _residual_from_q(self, q_flat: jnp.ndarray) -> jnp.ndarray:
        X = (self.nodes_flat + q_flat).reshape((self.mesh.num_nodes, 3))
        u = self._solve_release(X)
        return (q_flat.reshape((self.mesh.num_nodes, 3)) + u).reshape(-1)

    def _residual_from_c(self, c: jnp.ndarray) -> jnp.ndarray:
        return self._residual_from_q(self.psi @ c)

    def _projected_residual(self, c: jnp.ndarray) -> jnp.ndarray:
        residual = self._residual_from_c(c)
        return self.psi.T @ (self.weights * residual)

    def c0(self) -> np.ndarray:
        return np.zeros(self.mode_count, dtype=float)

    def q_from_c(self, c: np.ndarray) -> np.ndarray:
        return self.psi_np @ np.asarray(c, dtype=float)

    def residual(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self._residual_c_jit(jnp.asarray(c, dtype=jnp.float64)), dtype=float)

    def residual_from_q(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(self._residual_q_jit(jnp.asarray(q, dtype=jnp.float64)), dtype=float)

    def b(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self._b_jit(jnp.asarray(c, dtype=jnp.float64)), dtype=float)

    def J(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self._J_jit(jnp.asarray(c, dtype=jnp.float64)), dtype=float)

    def A(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self._A_jit(jnp.asarray(c, dtype=jnp.float64)), dtype=float)

    def J_fd(self, c: np.ndarray, h: float) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        J = np.zeros((self.mode_count, self.mode_count), dtype=float)
        for j in range(self.mode_count):
            step = np.zeros(self.mode_count, dtype=float)
            step[j] = h
            J[:, j] = (self.b(c + step) - self.b(c - step)) / (2.0 * h)
        return J

    def physical_norm(self, residual: np.ndarray) -> float:
        residual = np.asarray(residual, dtype=float)
        return float(np.sqrt(max(np.dot(self.weights_dof_np * residual, residual), 0.0)))

    def projected_norm(self, c: np.ndarray) -> float:
        return float(np.linalg.norm(self.b(c)))

    def bottom_violation(self, q: np.ndarray) -> Tuple[float, float]:
        values = np.asarray(q, dtype=float)[self.bottom_dof_mask_np]
        return float(np.max(np.abs(values))), float(np.linalg.norm(values))

    def audit_ad_fd(self) -> Dict[str, object]:
        c = self.c0()
        J_ad = self.J(c)
        best = None
        for h in FD_STEPS:
            J_fd = self.J_fd(c, h)
            diff = J_ad - J_fd
            column_den = np.maximum(np.linalg.norm(J_fd, axis=0), 1e-15)
            column_rel = np.linalg.norm(diff, axis=0) / column_den
            row = {
                "fd_step": float(h),
                "ad_fd_relative_error_F": float(
                    np.linalg.norm(diff, ord="fro") / max(np.linalg.norm(J_fd, ord="fro"), 1e-15)
                ),
                "ad_fd_max_abs_error": float(np.max(np.abs(diff))),
                "ad_fd_max_rel_column_error": float(np.max(column_rel)),
            }
            if best is None or row["ad_fd_relative_error_F"] < best["ad_fd_relative_error_F"]:
                best = row
        assert best is not None
        best["pass_fail"] = "PASS" if best["ad_fd_relative_error_F"] < 1e-5 else "FAIL"
        return best
