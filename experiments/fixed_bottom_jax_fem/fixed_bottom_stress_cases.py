"""Controlled hard response regimes for fixed-bottom stability diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from jax import config

config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from fixed_bottom_core import FixedBottomJaxResponse


@dataclass(frozen=True)
class HardCase:
    name: str
    geometry: str
    base_case: str
    matrix_kind: str
    diagonal_gain: float
    coupling_gain: float
    secondary_coupling: float
    nonlinear_beta: float
    trust_initial_lambda: float
    trust_max_step: float
    purpose: str


def hard_cases() -> List[HardCase]:
    return [
        HardCase(
            "high_gain_plate",
            "thin_plate",
            "plate_pure_bending",
            "diagonal_gain",
            2.75,
            0.12,
            0.0,
            0.0,
            1e-3,
            np.inf,
            "direct instability from modal gains above the unit-step stability range",
        ),
        HardCase(
            "coupled_plate_twist",
            "thin_plate",
            "plate_asymmetric_twist",
            "cyclic_coupling",
            1.05,
            1.65,
            0.35,
            0.0,
            2e-3,
            np.inf,
            "strong off-diagonal bending-to-twist response coupling",
        ),
        HardCase(
            "high_gain_comb_edge",
            "comb_coupon",
            "comb_edge_lift",
            "diagonal_gain",
            3.10,
            0.28,
            0.0,
            0.0,
            2e-3,
            np.inf,
            "finger-dependent gain mismatch causing direct modal divergence",
        ),
        HardCase(
            "coupled_comb_fingers",
            "comb_coupon",
            "comb_finger_varying_shrinkage",
            "cyclic_coupling",
            1.00,
            1.85,
            0.55,
            0.0,
            3e-3,
            np.inf,
            "strong finger-to-finger coupling that defeats diagonal calibration",
        ),
        HardCase(
            "nonlinear_step_invalid_comb",
            "comb_coupon",
            "comb_finger_varying_shrinkage",
            "nonlinear_coupling",
            1.20,
            1.10,
            0.30,
            400.0,
            2e-3,
            0.50,
            "smooth cubic modal response invalidates large undamped linearized steps",
        ),
    ]


def _cyclic_matrix(mode_count: int, shift: int) -> np.ndarray:
    matrix = np.zeros((mode_count, mode_count), dtype=float)
    for row in range(mode_count):
        matrix[row, (row + shift) % mode_count] = 1.0
    return matrix


def target_matrix(spec: HardCase, mode_count: int) -> np.ndarray:
    index = np.arange(mode_count, dtype=float)
    if spec.matrix_kind == "diagonal_gain":
        gains = spec.diagonal_gain * (0.82 + 0.36 * index / max(mode_count - 1, 1))
        matrix = np.diag(gains)
        matrix += spec.coupling_gain * _cyclic_matrix(mode_count, 1)
        return matrix

    p1 = _cyclic_matrix(mode_count, 1)
    p2 = _cyclic_matrix(mode_count, max(2, mode_count // 5))
    alternating = np.diag(np.where((index.astype(int) % 2) == 0, 1.0, -1.0))
    matrix = spec.diagonal_gain * np.eye(mode_count)
    matrix += spec.coupling_gain * p1
    matrix += spec.secondary_coupling * (alternating @ p2)
    return matrix


class HardResponseModel:
    """Duck-compatible response model with a controlled modal response layer."""

    def __init__(self, base: FixedBottomJaxResponse, spec: HardCase):
        self.base = base
        self.hard_spec = spec
        self.mesh = base.mesh
        self.mode_count = base.mode_count
        self.psi_np = base.psi_np
        self.weights_dof_np = base.weights_dof_np
        self.bottom_dof_mask_np = base.bottom_dof_mask_np
        self.case = spec
        self.matrix_np = target_matrix(spec, base.mode_count)
        self.matrix = jnp.asarray(self.matrix_np, dtype=jnp.float64)
        self.b0_np = base.b(base.c0())
        self.b0 = jnp.asarray(self.b0_np, dtype=jnp.float64)
        self.nonlinear_beta = float(spec.nonlinear_beta)

        rng = np.random.default_rng(20260625 + base.mode_count + len(spec.name))
        raw = rng.normal(size=(base.mode_count, base.mode_count))
        q, _ = np.linalg.qr(raw)
        self.nonlinear_rotation_np = q
        self.nonlinear_rotation = jnp.asarray(q, dtype=jnp.float64)

        self._b_jit = jax.jit(self._hard_b)
        self._J_jit = jax.jit(jax.jacfwd(self._hard_b))
        self._residual_jit = jax.jit(self._hard_residual)
        self._A_jit = jax.jit(jax.jacfwd(self._hard_residual))

    def _nonlinear(self, c: jnp.ndarray) -> jnp.ndarray:
        if self.nonlinear_beta == 0.0:
            return jnp.zeros_like(c)
        rotated = self.nonlinear_rotation @ c
        return self.nonlinear_beta * self.nonlinear_rotation.T @ (rotated**3)

    def _hard_b(self, c: jnp.ndarray) -> jnp.ndarray:
        return self.b0 + self.matrix @ c + self._nonlinear(c)

    def _hard_residual(self, c: jnp.ndarray) -> jnp.ndarray:
        base_residual = self.base._residual_c_jit(c)
        base_b = self.base._b_jit(c)
        correction = self.base.psi @ (self._hard_b(c) - base_b)
        return base_residual + correction

    def c0(self) -> np.ndarray:
        return self.base.c0()

    def q_from_c(self, c: np.ndarray) -> np.ndarray:
        return self.base.q_from_c(c)

    def residual(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self._residual_jit(jnp.asarray(c, dtype=jnp.float64)), dtype=float)

    def residual_from_q(self, q: np.ndarray) -> np.ndarray:
        return self.base.residual_from_q(q)

    def b(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self._b_jit(jnp.asarray(c, dtype=jnp.float64)), dtype=float)

    def J(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self._J_jit(jnp.asarray(c, dtype=jnp.float64)), dtype=float)

    def A(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(self._A_jit(jnp.asarray(c, dtype=jnp.float64)), dtype=float)

    def physical_norm(self, residual: np.ndarray) -> float:
        return self.base.physical_norm(residual)

    def projected_norm(self, c: np.ndarray) -> float:
        return float(np.linalg.norm(self.b(c)))

    def bottom_violation(self, q: np.ndarray):
        return self.base.bottom_violation(q)

    def J_fd(self, c: np.ndarray, h: float) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        matrix = np.zeros((self.mode_count, self.mode_count), dtype=float)
        for column in range(self.mode_count):
            step = np.zeros(self.mode_count, dtype=float)
            step[column] = h
            matrix[:, column] = (self.b(c + step) - self.b(c - step)) / (2.0 * h)
        return matrix
